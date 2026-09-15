// Download + load plumbing for both edge0 tiers.
//
// * 8B (`bailing_hybrid`) is the hand-written port in Edge0BailingHybrid.swift,
//   registered into mlx-swift-lm's model-type registry at startup.
// * 35B (`qwen3_5_moe`) needs no port — mlx-swift-lm already implements that
//   architecture natively; only edge0's LoRA has to be layered on top.
//
// Weights live in Application Support (not Caches — iOS purges Caches, and a
// 23 GB download is not something to re-fetch) with a Hugging Face-style
// snapshot layout managed by swift-huggingface's `HubCache`.

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Storage

enum Edge0Storage {
    /// Root of the on-device model store.
    static let modelsDirectory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first ?? FileManager.default.temporaryDirectory
        var url = base.appendingPathComponent("Edge0Models", isDirectory: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: true)
        }
        // Multi-GB checkpoints must never be uploaded to iCloud.
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try? url.setResourceValues(resourceValues)
        return url
    }()

    static var hubCache: HubCache { HubCache(cacheDirectory: modelsDirectory) }

    static func repoID(for tier: Edge0Tier) -> Repo.ID? {
        Repo.ID(rawValue: tier.repoId)
    }

    /// The downloaded snapshot for a tier, if one is present on disk.
    static func snapshotDirectory(for tier: Edge0Tier) -> URL? {
        guard let repo = repoID(for: tier) else { return nil }
        let snapshots = hubCache.snapshotsDirectory(repo: repo, kind: .model)
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: nil)
        else { return nil }
        // A snapshot counts as usable once its config.json is there.
        return entries.first { candidate in
            FileManager.default.fileExists(
                atPath: candidate.appendingPathComponent("config.json").path)
        }
    }

    static func isDownloaded(_ tier: Edge0Tier) -> Bool {
        guard let directory = snapshotDirectory(for: tier) else { return false }
        let contents =
            (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    /// Bytes occupied by a tier, following the blob store rather than the
    /// snapshot's symlinks.
    static func diskUsage(for tier: Edge0Tier) -> Int64 {
        guard let repo = repoID(for: tier) else { return 0 }
        return directorySize(hubCache.repoDirectory(repo: repo, kind: .model))
    }

    static func delete(_ tier: Edge0Tier) throws {
        guard let repo = repoID(for: tier) else { return }
        let directory = hubCache.repoDirectory(repo: repo, kind: .model)
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    static var freeDiskSpace: Int64 {
        let values = try? modelsDirectory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    private static func directorySize(_ url: URL) -> Int64 {
        guard
            let enumerator = FileManager.default.enumerator(
                at: url, includingPropertiesForKeys: [.fileAllocatedSizeKey, .isRegularFileKey],
                options: [])
        else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [
                .fileAllocatedSizeKey, .isRegularFileKey,
            ])
            guard values?.isRegularFile == true else { continue }
            total += Int64(values?.fileAllocatedSize ?? 0)
        }
        return total
    }
}

// MARK: - Downloader

/// Adapts swift-huggingface's `HubClient` to mlx-swift-lm's `Downloader`, with
/// the app's own cache directory rather than the macro's default.
struct Edge0Downloader: Downloader {
    private let client: HubClient

    init() {
        client = HubClient(cache: Edge0Storage.hubCache)
    }

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repo = Repo.ID(rawValue: id) else {
            throw Edge0LoadError.invalidRepositoryID(id)
        }
        return try await client.downloadSnapshot(
            of: repo,
            revision: revision ?? "main",
            matching: patterns,
            progressHandler: { @MainActor progress in progressHandler(progress) }
        )
    }
}

enum Edge0LoadError: LocalizedError {
    case invalidRepositoryID(String)
    case missingWeights(String)

    var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let id): "Geçersiz Hugging Face deposu: \(id)"
        case .missingWeights(let path): "Model dosyaları bulunamadı: \(path)"
        }
    }
}

// MARK: - Loading

struct Edge0LoadedModel {
    let tier: Edge0Tier
    let container: ModelContainer
    let directory: URL
    let loraReport: Edge0LoRAReport?
    let parameterCount: Int
}

enum Edge0Loader {
    /// Teaches mlx-swift-lm about the 8B tier's architecture. The 35B tier's
    /// `qwen3_5_moe` is already in the stock registry. Registering the same
    /// type twice just replaces an identical creator, so this needs no
    /// "already done" flag (and therefore has nothing to race on).
    static func registerModelTypes() async {
        await LLMTypeRegistry.shared.registerModelType(Edge0Tier.edge0_8b.modelType) { data in
            let configuration = try JSONDecoder().decode(
                Edge0BailingConfiguration.self, from: data)
            return Edge0BailingModel(configuration)
        }
    }

    static func load(
        tier: Edge0Tier,
        applyLoRA: Bool,
        gpuCacheLimitMB: Int,
        hotExpertSlots: Int,
        streamExperts: Bool,
        onProgress: @escaping @Sendable (Progress) -> Void
    ) async throws -> Edge0LoadedModel {
        await registerModelTypes()

        // A phone has no memory to spare for MLX's buffer cache.
        MLX.GPU.set(cacheLimit: gpuCacheLimitMB * 1024 * 1024)

        let configuration = ModelConfiguration(
            id: tier.repoId, eosTokenIds: tier.eosTokenIds)

        let resolved = try await resolve(
            configuration: configuration, from: Edge0Downloader(), useLatest: false,
            progressHandler: onProgress)

        let loraURL = resolved.modelDirectory.appendingPathComponent(tier.loraFileName)
        let container: ModelContainer
        var report: Edge0LoRAReport?

        if streamExperts || tier.requiresExpertStreaming {
            // The 35B checkpoint is far larger than any phone's memory, so its
            // experts stay on disk and are read per step.
            let loaded = try await Edge0StreamingLoader.load(
                tier: tier,
                directory: resolved.modelDirectory,
                tokenizerLoader: #huggingFaceTokenizerLoader(),
                hotSlotsPerLayer: hotExpertSlots,
                loraURL: applyLoRA ? loraURL : nil
            )
            container = ModelContainer(context: loaded.context)
            report = loaded.loraReport
        } else {
            container = try await LLMModelFactory.shared.loadContainer(
                from: resolved.modelDirectory,
                using: #huggingFaceTokenizerLoader())

            if applyLoRA, FileManager.default.fileExists(atPath: loraURL.path) {
                report = try await container.perform { context in
                    try Edge0LoRA.apply(
                        to: context.model, fileURL: loraURL, rank: 16, alpha: 32.0)
                }
            }
        }

        let parameterCount = await container.perform { context in
            context.model.numParameters()
        }

        return Edge0LoadedModel(
            tier: tier,
            container: container,
            directory: resolved.modelDirectory,
            loraReport: report,
            parameterCount: parameterCount
        )
    }
}
