import Darwin
import Foundation
import MLX
import MLXLMCommon
import UIKit

/// Owns model lifecycle: what is on disk, what is loaded, and the progress of
/// getting from one to the other. The views observe this; nothing else talks to
/// `Edge0Loader` directly.
@Observable
@MainActor
final class ModelManager {
    enum Phase: Equatable {
        case idle
        case downloading(fraction: Double, detail: String)
        case preparing(String)
        case ready
        case failed(String)

        var isBusy: Bool {
            switch self {
            case .downloading, .preparing: true
            case .idle, .ready, .failed: false
            }
        }
    }

    private(set) var phase: Phase = .idle
    private(set) var activeTier: Edge0Tier?
    /// The tier a download/load is currently working on.
    private(set) var pendingTier: Edge0Tier?
    private(set) var loaded: Edge0LoadedModel?
    private(set) var downloadedTiers: Set<Edge0Tier> = []
    private(set) var diskUsage: [Edge0Tier: Int64] = [:]
    private(set) var freeDiskSpace: Int64 = 0

    private var loadTask: Task<Void, Never>?
    /// Stamps each load so a superseded one cannot report progress, publish a
    /// result, or clear its successor's task handle.
    private var loadGeneration = 0
    /// Last progress sample and a smoothed rate, for the download ETA.
    private var lastProgressSample: (at: Date, bytes: Int64)?
    private var smoothedBytesPerSecond: Double = 0
    /// `deinit` is nonisolated, so the token it has to release cannot be
    /// main-actor state. Only `init` and `deinit` touch it.
    nonisolated(unsafe) private var memoryWarningObserver: (any NSObjectProtocol)?

    init() {
        refreshStorage()
        observeMemoryPressure()
    }

    deinit {
        if let memoryWarningObserver {
            NotificationCenter.default.removeObserver(memoryWarningObserver)
        }
    }

    // MARK: Memory pressure

    /// When iOS warns, the app has seconds to hand memory back before it is
    /// killed. The streamed expert caches are the largest thing that can be
    /// dropped without losing the model: the next step pages those weights
    /// back in from the mapping, so the answer is unaffected, only slower.
    private func observeMemoryPressure() {
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main
        ) { _ in
            // Logged because a purge is invisible from the outside and looks
            // exactly like the model having got slower for no reason: every
            // expert is a miss again until the caches refill. If these show up
            // in a run, the cache budget is the thing to look at, not the code.
            Edge0Log.memory("bellek uyarısı — expert önbellekleri boşaltılıyor")
            MLX.GPU.clearCache()
            Edge0ExpertCaches.purge()
        }
    }

    // MARK: Storage

    func refreshStorage() {
        var downloaded: Set<Edge0Tier> = []
        var usage: [Edge0Tier: Int64] = [:]
        for tier in Edge0Tier.allCases {
            if Edge0Storage.isDownloaded(tier) { downloaded.insert(tier) }
            usage[tier] = Edge0Storage.diskUsage(for: tier)
        }
        downloadedTiers = downloaded
        diskUsage = usage
        freeDiskSpace = Edge0Storage.freeDiskSpace
        for tier in Edge0Tier.allCases {
            Edge0Log.write(
                "depo \(tier.rawValue): \(Edge0Storage.localDirectory(for: tier)?.path ?? "yok")")
        }
    }

    func delete(tier: Edge0Tier) {
        if activeTier == tier { unload() }
        try? Edge0Storage.delete(tier)
        refreshStorage()
    }

    /// True when the tier's expected peak fits inside what iOS will let this
    /// process allocate. False is a warning, not a block: the budget moves
    /// around, and the expert cache can be turned down to make room.
    func hasMemoryHeadroom(for tier: Edge0Tier) -> Bool {
        let available = Self.availableProcessMemoryBytes
        guard available > 0 else { return true }
        return Double(available) > tier.peakActiveMemoryGB * 1_073_741_824
    }

    /// True when the tier's checkpoint plausibly fits the device's free space.
    func hasRoom(for tier: Edge0Tier) -> Bool {
        if downloadedTiers.contains(tier) { return true }
        let needed = Int64(tier.downloadSizeGB * 1.08 * 1_073_741_824)
        return freeDiskSpace > needed
    }

    // MARK: Lifecycle

    func prepare(tier: Edge0Tier, settings: AppSettings) {
        guard loadTask == nil else { return }
        if activeTier == tier, loaded != nil, phase == .ready { return }

        unload()
        pendingTier = tier
        lastProgressSample = nil
        smoothedBytesPerSecond = 0
        Edge0Log.write(
            "hazırlanıyor: \(tier.rawValue) · LoRA \(settings.useLoRA)"
                + " · akış \(settings.expertStreaming)"
                + " · prerouter \(settings.usePrerouter)"
                + " · önbellek \(settings.expertCacheBudgetMB) MB")
        Edge0Log.memory("yükleme öncesi")
        phase = .downloading(fraction: 0, detail: "Bağlanılıyor…")
        // A multi-GB download dies if the screen locks and the app suspends.
        UIApplication.shared.isIdleTimerDisabled = true

        // A cancelled load finishes asynchronously, so by the time its tail
        // runs a newer one may already be in flight. Without this stamp the
        // old task clears the new task's handle on its way out, and the guard
        // at the top of this method then lets a second load start alongside
        // it — two checkpoints resident at once, on a phone.
        loadGeneration += 1
        let generation = loadGeneration

        loadTask = Task { [weak self] in
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                if self?.loadGeneration == generation { self?.loadTask = nil }
            }
            do {
                let result = try await Edge0Loader.load(
                    tier: tier,
                    applyLoRA: settings.useLoRA,
                    gpuCacheLimitMB: settings.gpuCacheLimitMB,
                    expertCacheBudgetMB: settings.expertCacheBudgetMB,
                    streamExperts: settings.expertStreaming,
                    usePrerouter: settings.usePrerouter,
                    onProgress: { progress in
                        Task { @MainActor [weak self] in
                            guard self?.loadGeneration == generation else { return }
                            self?.report(progress)
                        }
                    },
                    onRetry: { attempt, total, error in
                        Task { @MainActor [weak self] in
                            guard self?.loadGeneration == generation else { return }
                            self?.reportRetry(attempt: attempt, of: total, error: error)
                        }
                    }
                )
                guard let self, !Task.isCancelled, self.loadGeneration == generation else {
                    return
                }
                self.loaded = result
                self.activeTier = tier
                self.pendingTier = nil
                self.phase = .ready
                self.refreshStorage()
                Edge0Log.write(
                    "yüklendi: \(tier.rawValue) · \(result.parameterCount) parametre"
                        + " · sağlık: \(result.health.detail)")
                if !result.health.sample.isEmpty {
                    Edge0Log.write("açılış örneği: \(result.health.sample)")
                }
                Edge0Log.memory("yükleme sonrası")
                // Survived: the next launch may auto-load again.
                Edge0SafeBoot.clear()
            } catch is CancellationError {
                guard self?.loadGeneration == generation else { return }
                Edge0Log.write("yükleme iptal edildi: \(tier.rawValue)")
                Edge0SafeBoot.clear()
                self?.phase = .idle
            } catch {
                guard self?.loadGeneration == generation else { return }
                Edge0Log.failure("yükleme \(tier.rawValue)", error)
                // A clean failure is not a crash, so it must not disarm the
                // next auto-load — the error is on screen to act on.
                Edge0SafeBoot.clear()
                self?.phase = .failed(error.localizedDescription)
            }
        }
    }

    /// Surfaced when safe boot skipped the automatic load.
    func reportAutoLoadSkipped() {
        phase = .failed(
            "Önceki açılışta model yüklenirken uygulama kapandı, bu yüzden otomatik"
                + " yükleme atlandı. Yüklemeyi kendin başlatabilir ya da Ayarlar'dan"
                + " otomatik yüklemeyi kapatabilirsin.")
    }

    /// Copies a checkpoint the user already has into the app's storage, then
    /// loads it. Reuses the load stamp and phase machinery so an import cannot
    /// race a download, and so cancelling works the same way.
    func importModel(tier: Edge0Tier, from sources: [URL], settings: AppSettings) {
        guard !sources.isEmpty else {
            phase = .failed(Edge0ImportError.nothingSelected.localizedDescription)
            pendingTier = tier
            return
        }
        // An import is an explicit instruction, so it takes over from whatever
        // was running rather than returning in silence — which looked exactly
        // like the button doing nothing.
        unload()
        pendingTier = tier
        lastProgressSample = nil
        smoothedBytesPerSecond = 0
        phase = .downloading(fraction: 0, detail: "Kopyalanıyor…")
        UIApplication.shared.isIdleTimerDisabled = true

        loadGeneration += 1
        let generation = loadGeneration

        loadTask = Task { [weak self] in
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                if self?.loadGeneration == generation { self?.loadTask = nil }
            }
            do {
                try await Task.detached(priority: .utility) {
                    _ = try Edge0Importer.run(tier: tier, from: sources) { copied, total in
                        Task { @MainActor [weak self] in
                            guard self?.loadGeneration == generation else { return }
                            self?.reportCopy(copied: copied, total: total)
                        }
                    }
                }.value

                guard let self, !Task.isCancelled, self.loadGeneration == generation else {
                    return
                }
                self.refreshStorage()
                self.loadTask = nil
                self.prepare(tier: tier, settings: settings)
            } catch is CancellationError {
                guard self?.loadGeneration == generation else { return }
                self?.phase = .idle
            } catch {
                guard self?.loadGeneration == generation else { return }
                self?.phase = .failed(error.localizedDescription)
            }
        }
    }

    private func reportCopy(copied: Int64, total: Int64) {
        guard total > 0 else { return }
        let fraction = Double(copied) / Double(total)
        phase =
            fraction >= 1.0
            ? .preparing("Ağırlıklar yükleniyor…")
            : .downloading(
                fraction: fraction,
                detail: describeTransfer(completed: copied, total: total))
    }

    func cancelLoad() {
        // Bumping the stamp is what makes this stick: cancellation is
        // cooperative, so the task may still be mid-load and would otherwise
        // publish its result over the state the user just asked for.
        loadGeneration += 1
        loadTask?.cancel()
        loadTask = nil
        pendingTier = nil
        phase = .idle
    }

    func unload() {
        loadGeneration += 1
        loadTask?.cancel()
        loadTask = nil
        loaded = nil
        activeTier = nil
        pendingTier = nil
        phase = .idle
        MLX.GPU.clearCache()
    }

    /// Builds a fresh chat session against the loaded model.
    ///
    /// `history` re-hydrates a transcript: the session prefills those turns on
    /// its next response, so resuming a saved conversation gives the model the
    /// same context it had when the conversation was live. The system prompt is
    /// passed separately and must not appear in `history`.
    func makeSession(settings: AppSettings, history: [Chat.Message] = []) -> ChatSession? {
        guard let loaded else { return nil }
        let instructions = settings.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let system: String? = instructions.isEmpty ? nil : instructions

        if history.isEmpty {
            return ChatSession(
                loaded.container,
                instructions: system,
                generateParameters: settings.generateParameters,
                additionalContext: ["enable_thinking": settings.thinkingMode]
            )
        }

        return ChatSession(
            loaded.container,
            instructions: system,
            history: history,
            generateParameters: settings.generateParameters,
            additionalContext: ["enable_thinking": settings.thinkingMode]
        )
    }

    private func report(_ progress: Progress) {
        let fraction = progress.fractionCompleted
        let total = progress.totalUnitCount
        let detail: String

        if total > 0, total < 200 {
            // Small unit counts mean the downloader is counting files.
            detail = "Dosya \(progress.completedUnitCount + 1)/\(total)"
        } else if total > 0 {
            // NOT `completedUnitCount`: a Progress with children only advances
            // that when a whole child finishes, so on a multi-gigabyte shard it
            // sits still for minutes while `fractionCompleted` keeps moving.
            // Reading bytes off the frozen counter made a working download
            // report 0 MB/s and an ETA of hundreds of hours.
            let completed = Int64(fraction * Double(total))
            detail = describeTransfer(completed: completed, total: total)
        } else {
            detail = "İndiriliyor…"
        }

        if fraction >= 1.0 {
            phase = .preparing("Ağırlıklar yükleniyor…")
        } else {
            phase = .downloading(fraction: fraction, detail: detail)
        }
    }

    /// A dropped connection on a multi-hour download is normal and recoverable,
    /// so it is reported as what it is rather than left looking like a stall.
    private func reportRetry(attempt: Int, of total: Int, error: Error) {
        guard case .downloading(let fraction, _) = phase else { return }
        // The transfer restarts from the partial file, so the rate baseline
        // from before the drop would produce nonsense.
        lastProgressSample = nil
        smoothedBytesPerSecond = 0
        phase = .downloading(
            fraction: fraction,
            detail: "Bağlantı koptu — yeniden deneniyor (\(attempt)/\(total - 1))…")
    }

    /// "3,2 GB / 23 GB · 11 MB/sn · ~34 dk", with the rate and estimate left
    /// off until they mean something.
    private func describeTransfer(completed: Int64, total: Int64) -> String {
        var parts = ["\(Self.formatBytes(completed)) / \(Self.formatBytes(total))"]
        guard let rate = updateRate(completed: completed), rate > 1024 else {
            return parts.joined(separator: " · ")
        }
        parts.append("\(Self.formatBytes(Int64(rate)))/sn")
        let remaining = Double(total - completed) / rate
        // A wild estimate is worse than none: it reads as "this is broken".
        if remaining.isFinite, remaining > 0, remaining < 24 * 3600 {
            parts.append("~\(Self.formatDuration(remaining))")
        }
        return parts.joined(separator: " · ")
    }

    /// Exponentially smoothed download rate in bytes per second. Raw samples
    /// swing wildly as files start and finish, which makes an ETA built on
    /// them jump around uselessly.
    private func updateRate(completed: Int64) -> Double? {
        let now = Date()
        guard let previous = lastProgressSample else {
            lastProgressSample = (now, completed)
            return nil
        }

        let elapsed = now.timeIntervalSince(previous.at)
        let delta = completed - previous.bytes
        // The baseline only advances when a sample is actually taken.
        // Resetting it on every call would keep the window under the
        // threshold forever and no rate would ever be computed.
        guard elapsed > 0.5, delta >= 0 else {
            return smoothedBytesPerSecond > 0 ? smoothedBytesPerSecond : nil
        }
        lastProgressSample = (now, completed)

        let sample = Double(delta) / elapsed
        smoothedBytesPerSecond =
            smoothedBytesPerSecond == 0
            ? sample
            : smoothedBytesPerSecond * 0.8 + sample * 0.2
        return smoothedBytesPerSecond
    }

    /// Pure formatters, so they stay callable from error descriptions and
    /// other nonisolated contexts.
    nonisolated static func formatDuration(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total < 60 { return "\(total) sn" }
        let minutes = total / 60
        if minutes < 60 { return "\(minutes) dk" }
        let hours = minutes / 60
        let remainder = minutes % 60
        return remainder == 0 ? "\(hours) sa" : "\(hours) sa \(remainder) dk"
    }

    // MARK: Memory

    static var physicalMemoryBytes: Int64 { Int64(ProcessInfo.processInfo.physicalMemory) }

    /// What iOS will actually let *this process* allocate before jetsam kills
    /// it. This is the number that decides whether a tier fits, not the
    /// device's RAM: an app gets a fraction of the latter, and how big a
    /// fraction depends on entitlements and on what else the phone is doing.
    static var availableProcessMemoryBytes: Int64 { Int64(os_proc_available_memory()) }

    static var mlxActiveMemoryBytes: Int64 { Int64(MLX.GPU.activeMemory) }

    static var mlxPeakMemoryBytes: Int64 { Int64(MLX.GPU.peakMemory) }

    nonisolated static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
