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
    /// Root of the on-device model store, inside Documents.
    ///
    /// Documents rather than Application Support because the app declares
    /// `UIFileSharingEnabled`: this is the one directory that shows up in the
    /// Files app and in Finder, which is what makes a checkpoint something the
    /// user can put there, inspect and delete themselves.
    static let modelsDirectory: URL = {
        let manager = FileManager.default
        let base = manager.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? manager.temporaryDirectory
        var url = base.appendingPathComponent("Models", isDirectory: true)

        migrateFromApplicationSupport(into: url)

        if !manager.fileExists(atPath: url.path) {
            try? manager.createDirectory(at: url, withIntermediateDirectories: true)
        }
        // Documents is backed up by default, and a 23 GB checkpoint that can be
        // downloaded again has no business in an iCloud backup.
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try? url.setResourceValues(resourceValues)

        // Created eagerly so that someone opening the app's folder in Files
        // sees where a checkpoint is meant to go instead of an empty directory.
        for tier in Edge0Tier.allCases {
            try? manager.createDirectory(
                at: url.appendingPathComponent(tier.rawValue, isDirectory: true),
                withIntermediateDirectories: true)
        }
        return url
    }()

    /// Moves a store written by an earlier build, so an interrupted 23 GB
    /// download is not silently orphaned by the change of location. Within one
    /// volume this is a rename, so its cost does not depend on the size.
    private static func migrateFromApplicationSupport(into destination: URL) {
        let manager = FileManager.default
        guard
            let support = manager.urls(for: .applicationSupportDirectory, in: .userDomainMask)
                .first
        else { return }
        let old = support.appendingPathComponent("Edge0Models", isDirectory: true)
        guard manager.fileExists(atPath: old.path) else { return }

        if !manager.fileExists(atPath: destination.path) {
            try? manager.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if (try? manager.moveItem(at: old, to: destination)) != nil { return }
        }

        // Destination already there: take across whatever does not collide,
        // then drop the old directory if it emptied out.
        let names = (try? manager.contentsOfDirectory(atPath: old.path)) ?? []
        for name in names {
            let target = destination.appendingPathComponent(name)
            guard !manager.fileExists(atPath: target.path) else { continue }
            try? manager.moveItem(at: old.appendingPathComponent(name), to: target)
        }
        if (try? manager.contentsOfDirectory(atPath: old.path))?.isEmpty == true {
            try? manager.removeItem(at: old)
        }
    }

    /// Downloads go to their own subdirectory so the Hub's blob store does not
    /// clutter the folders the user is meant to drop files into.
    static var hubCacheDirectory: URL {
        let url = modelsDirectory.appendingPathComponent("hub", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static var hubCache: HubCache { HubCache(cacheDirectory: hubCacheDirectory) }

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

    /// Where a tier's own files live: imported through the picker, or dropped
    /// straight in from Finder or the Files app. One folder per tier, named
    /// after it, directly under the visible root — the shallower the path, the
    /// likelier someone puts the files in the right place.
    static func importedDirectory(for tier: Edge0Tier) -> URL {
        modelsDirectory.appendingPathComponent(tier.rawValue, isDirectory: true)
    }

    /// The path to show the user, as it reads in the Files app.
    static func displayPath(for tier: Edge0Tier) -> String {
        "Edge0 Demo/Models/\(tier.rawValue)"
    }

    /// True when a directory holds something the loaders can actually use.
    private static func isUsable(_ directory: URL) -> Bool {
        guard
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("config.json").path)
        else { return false }
        let contents =
            (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    /// A usable checkpoint at `directory`, or in one of its immediate
    /// subfolders.
    ///
    /// Dragging the downloaded folder into the tier's folder, rather than its
    /// contents, is the obvious thing to do and leaves the files one level
    /// deeper than the loader looks. Checking one level down costs a directory
    /// listing and saves a correct-looking drop from reading as "nothing
    /// happened".
    private static func usableDirectory(at directory: URL) -> URL? {
        if isUsable(directory) { return directory }
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey])
        else { return nil }
        return entries.first { entry in
            (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
                && isUsable(entry)
        }
    }

    /// The checkpoint to load, wherever it came from. An imported copy wins:
    /// the user put it there deliberately.
    static func localDirectory(for tier: Edge0Tier) -> URL? {
        if let imported = usableDirectory(at: importedDirectory(for: tier)) { return imported }
        if let snapshot = snapshotDirectory(for: tier), isUsable(snapshot) { return snapshot }
        return nil
    }

    static func isDownloaded(_ tier: Edge0Tier) -> Bool {
        localDirectory(for: tier) != nil
    }

    /// True when the tier's files were imported rather than downloaded.
    static func isImported(_ tier: Edge0Tier) -> Bool {
        usableDirectory(at: importedDirectory(for: tier)) != nil
    }

    /// Bytes occupied by a tier, following the blob store rather than the
    /// snapshot's symlinks.
    static func diskUsage(for tier: Edge0Tier) -> Int64 {
        var total = directorySize(importedDirectory(for: tier))
        if let repo = repoID(for: tier) {
            total += directorySize(hubCache.repoDirectory(repo: repo, kind: .model))
        }
        return total
    }

    static func delete(_ tier: Edge0Tier) throws {
        let imported = importedDirectory(for: tier)
        if FileManager.default.fileExists(atPath: imported.path) {
            try FileManager.default.removeItem(at: imported)
        }
        // Put the empty folder back: it is the signpost for where to drop
        // files, and it disappearing from Files after a delete is confusing.
        try? FileManager.default.createDirectory(
            at: imported, withIntermediateDirectories: true)
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
    private let onRetry: @Sendable (Int, Int, Error) -> Void

    /// Attempts in total, not retries after the first.
    private static let attempts = 5

    init(onRetry: @escaping @Sendable (Int, Int, Error) -> Void = { _, _, _ in }) {
        client = HubClient(cache: Edge0Storage.hubCache)
        self.onRetry = onRetry
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

        // A 23 GB download over a phone's connection will be interrupted; on a
        // long enough transfer that is a certainty, not an edge case. Hugging
        // Face downloads here resume from the partial blob, so a retry picks up
        // where the last attempt stopped rather than starting over — which
        // makes giving up on the first dropped connection simply wrong.
        var lastError: Error?
        for attempt in 1 ... Self.attempts {
            do {
                return try await client.downloadSnapshot(
                    of: repo,
                    revision: revision ?? "main",
                    matching: patterns,
                    progressHandler: { @MainActor progress in progressHandler(progress) }
                )
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // The user asking to stop must not look like a network failure.
                try Task.checkCancellation()
                lastError = error
                guard attempt < Self.attempts else { break }
                onRetry(attempt, Self.attempts, error)
                // 2s, 4s, 8s, 16s: long enough to outlast a handover between
                // cells or a Wi-Fi reconnect.
                try await Task.sleep(for: .seconds(pow(2.0, Double(attempt))))
            }
        }
        throw lastError ?? Edge0LoadError.invalidRepositoryID(id)
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
    let health: Edge0HealthReport
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
        expertCacheBudgetMB: Int,
        streamExperts: Bool,
        onProgress: @escaping @Sendable (Progress) -> Void,
        onRetry: @escaping @Sendable (Int, Int, Error) -> Void = { _, _, _ in }
    ) async throws -> Edge0LoadedModel {
        await registerModelTypes()

        // A phone has no memory to spare for MLX's buffer cache.
        MLX.GPU.set(cacheLimit: gpuCacheLimitMB * 1024 * 1024)

        // Already on the device — imported, or downloaded on an earlier run —
        // so there is nothing to fetch and nothing to check against the Hub.
        let directory: URL
        if let local = Edge0Storage.localDirectory(for: tier) {
            directory = local
        } else {
            let configuration = ModelConfiguration(
                id: tier.repoId, eosTokenIds: tier.eosTokenIds)
            let resolved = try await resolve(
                configuration: configuration, from: Edge0Downloader(onRetry: onRetry),
                useLatest: false, progressHandler: onProgress)
            directory = resolved.modelDirectory
        }

        let loraURL = directory.appendingPathComponent(tier.loraFileName)
        let container: ModelContainer
        var report: Edge0LoRAReport?

        if tier.supportsExpertStreaming, streamExperts || tier.requiresExpertStreaming {
            // The 35B checkpoint is far larger than any phone's memory, so its
            // experts stay on disk and are read per step.
            let loaded = try await Edge0StreamingLoader.load(
                tier: tier,
                directory: directory,
                tokenizerLoader: #huggingFaceTokenizerLoader(),
                expertCacheBudgetBytes: expertCacheBudgetMB * 1024 * 1024,
                loraURL: applyLoRA ? loraURL : nil
            )
            container = ModelContainer(context: loaded.context)
            report = loaded.loraReport
        } else {
            container = try await LLMModelFactory.shared.loadContainer(
                from: directory,
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

        // One token through the model: a shape or dtype slip in the port shows
        // up here as NaN logits instead of as gibberish an hour later.
        let health = await container.perform { context in
            Edge0ModelHealth.check(model: context.model, tokenizer: context.tokenizer)
        }
        guard health.passed else { throw Edge0HealthError.failed(health.detail) }

        return Edge0LoadedModel(
            tier: tier,
            container: container,
            directory: directory,
            loraReport: report,
            parameterCount: parameterCount,
            health: health
        )
    }
}
