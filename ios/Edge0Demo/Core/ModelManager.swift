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
        phase = .downloading(fraction: 0, detail: "Bağlanılıyor…")
        // A multi-GB download dies if the screen locks and the app suspends.
        UIApplication.shared.isIdleTimerDisabled = true

        loadTask = Task { [weak self] in
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                self?.loadTask = nil
            }
            do {
                let result = try await Edge0Loader.load(
                    tier: tier,
                    applyLoRA: settings.useLoRA,
                    gpuCacheLimitMB: settings.gpuCacheLimitMB,
                    expertCacheBudgetMB: settings.expertCacheBudgetMB,
                    streamExperts: settings.expertStreaming,
                    onProgress: { progress in
                        Task { @MainActor [weak self] in
                            self?.report(progress)
                        }
                    }
                )
                guard let self, !Task.isCancelled else { return }
                self.loaded = result
                self.activeTier = tier
                self.pendingTier = nil
                self.phase = .ready
                self.refreshStorage()
            } catch is CancellationError {
                self?.phase = .idle
            } catch {
                self?.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancelLoad() {
        loadTask?.cancel()
        loadTask = nil
        pendingTier = nil
        phase = .idle
    }

    func unload() {
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
        let detail: String
        if progress.totalUnitCount > 0, progress.totalUnitCount < 200 {
            detail = "Dosya \(progress.completedUnitCount + 1)/\(progress.totalUnitCount)"
        } else if progress.totalUnitCount > 0 {
            detail =
                "\(Self.formatBytes(progress.completedUnitCount)) / \(Self.formatBytes(progress.totalUnitCount))"
        } else {
            detail = "İndiriliyor…"
        }
        if fraction >= 1.0 {
            phase = .preparing("Ağırlıklar yükleniyor…")
        } else {
            phase = .downloading(fraction: fraction, detail: detail)
        }
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

    static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
