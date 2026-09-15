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

    init() {
        refreshStorage()
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
    func makeSession(settings: AppSettings) -> ChatSession? {
        guard let loaded else { return nil }
        let instructions = settings.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return ChatSession(
            loaded.container,
            instructions: instructions.isEmpty ? nil : instructions,
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

    static var mlxActiveMemoryBytes: Int64 { Int64(MLX.GPU.activeMemory) }

    static var mlxPeakMemoryBytes: Int64 { Int64(MLX.GPU.peakMemory) }

    static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
