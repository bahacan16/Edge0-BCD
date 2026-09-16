// Importing a checkpoint that was fetched somewhere else.
//
// The app can download both tiers itself, but the 35B checkpoint is ~23 GB.
// Anyone who already has it — on a Mac, in iCloud Drive, on a USB-C drive —
// should not have to pull it again over the phone's connection, so a folder
// can be handed to the app directly instead.
//
// The copy is chunked rather than a single `copyItem` so that a multi-hour
// import can report progress and be cancelled.

import Foundation

enum Edge0ImportError: LocalizedError {
    case notReadable
    case missingConfig
    case noWeights
    case notEnoughSpace(needed: Int64, free: Int64)

    var errorDescription: String? {
        switch self {
        case .notReadable:
            "Seçilen klasör okunamadı."
        case .missingConfig:
            "Klasörde config.json yok — model klasörünün kendisini seçin."
        case .noWeights:
            "Klasörde model ağırlığı (model-*.safetensors) yok."
        case .notEnoughSpace(let needed, let free):
            """
            Yeterli yer yok: \(ModelManager.formatBytes(needed)) gerekiyor, \
            \(ModelManager.formatBytes(free)) boş.
            """
        }
    }
}

enum Edge0Importer {
    /// Files the loaders actually read.
    ///
    /// `prerouter_*` is deliberately left out: this app does not implement
    /// edge0's prerouter (it is a streaming prefetch optimisation, a no-op on
    /// the routing itself), and the file is hundreds of megabytes.
    static func shouldImport(_ name: String) -> Bool {
        guard !name.hasPrefix("."), !name.hasPrefix("prerouter_") else { return false }
        return name.hasSuffix(".safetensors") || name.hasSuffix(".json")
            || name.hasSuffix(".jinja") || name.hasSuffix(".model") || name.hasSuffix(".txt")
    }

    /// Copies the importable files of `source` into the tier's own directory.
    ///
    /// `source` is expected to be security-scoped; the caller opens and closes
    /// that access around this call.
    static func run(
        tier: Edge0Tier,
        from source: URL,
        onProgress: @Sendable @escaping (_ copied: Int64, _ total: Int64) -> Void
    ) throws -> URL {
        let manager = FileManager.default
        guard
            let names = try? manager.contentsOfDirectory(atPath: source.path)
        else { throw Edge0ImportError.notReadable }

        let wanted = names.filter(shouldImport).sorted()
        guard wanted.contains("config.json") else { throw Edge0ImportError.missingConfig }
        guard wanted.contains(where: { $0.hasSuffix(".safetensors") && $0.hasPrefix("model") })
        else { throw Edge0ImportError.noWeights }

        var total: Int64 = 0
        for name in wanted {
            let values = try? source.appendingPathComponent(name)
                .resourceValues(forKeys: [.fileSizeKey])
            total += Int64(values?.fileSize ?? 0)
        }

        let free = Edge0Storage.freeDiskSpace
        guard free > total else {
            throw Edge0ImportError.notEnoughSpace(needed: total, free: free)
        }

        // Import into a staging directory and move it into place at the end,
        // so a cancelled or failed import never leaves something that looks
        // like a usable model.
        let destination = Edge0Storage.importedDirectory(for: tier)
        let staging = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(tier.rawValue).partial", isDirectory: true)
        try? manager.removeItem(at: staging)
        try manager.createDirectory(at: staging, withIntermediateDirectories: true)

        var copied: Int64 = 0
        do {
            for name in wanted {
                try Task.checkCancellation()
                try copy(
                    from: source.appendingPathComponent(name),
                    to: staging.appendingPathComponent(name),
                    copied: &copied, total: total, onProgress: onProgress)
            }
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }

        try? manager.removeItem(at: destination)
        try manager.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try manager.moveItem(at: staging, to: destination)
        return destination
    }

    /// Chunked so progress advances within a single multi-gigabyte shard and
    /// cancellation is noticed promptly.
    private static func copy(
        from source: URL, to destination: URL,
        copied: inout Int64, total: Int64,
        onProgress: @Sendable (Int64, Int64) -> Void
    ) throws {
        let manager = FileManager.default
        guard let input = try? FileHandle(forReadingFrom: source) else {
            throw Edge0ImportError.notReadable
        }
        defer { try? input.close() }

        manager.createFile(atPath: destination.path, contents: nil)
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }

        let chunkSize = 8 * 1024 * 1024
        while true {
            try Task.checkCancellation()
            guard let chunk = try input.read(upToCount: chunkSize), !chunk.isEmpty else { break }
            try output.write(contentsOf: chunk)
            copied += Int64(chunk.count)
            onProgress(copied, total)
        }
    }
}
