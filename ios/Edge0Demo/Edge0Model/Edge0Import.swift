// Importing a checkpoint that was fetched somewhere else.
//
// The app can download both tiers itself, but the 35B checkpoint is ~23 GB.
// Anyone who already has it — on a Mac, in iCloud Drive, on a USB-C drive —
// should not have to pull it again over the phone's connection, so files can
// be handed to the app directly instead.
//
// Both a folder and a straight multi-selection of files are accepted. Picking
// the files is the more reliable route: folder selection depends on the Files
// picker granting scoped access to a directory and on every item inside it
// being materialised, and it fails in ways that look like the picker simply
// ignoring the Open button.
//
// The copy is chunked rather than a single `copyItem` so that a multi-hour
// import reports progress and can be cancelled.

import Foundation

enum Edge0ImportError: LocalizedError {
    case nothingSelected
    case notReadable(String)
    case missingConfig
    case noWeights
    case notEnoughSpace(needed: Int64, free: Int64)
    case notDownloaded(String)

    var errorDescription: String? {
        switch self {
        case .nothingSelected:
            "Hiçbir dosya seçilmedi."
        case .notReadable(let name):
            "Okunamadı: \(name). Klasör yerine dosyaları seçmeyi deneyin."
        case .missingConfig:
            """
            Seçimde config.json yok. Klasörün içine girip tüm dosyaları seçin, \
            ya da dosyaları Dosyalar uygulamasından doğrudan uygulamanın model \
            klasörüne kopyalayın.
            """
        case .noWeights:
            "Seçimde model ağırlığı (model-*.safetensors) yok."
        case .notEnoughSpace(let needed, let free):
            """
            Yeterli yer yok: \(ModelManager.formatBytes(needed)) gerekiyor, \
            \(ModelManager.formatBytes(free)) boş.
            """
        case .notDownloaded(let name):
            """
            \(name) henüz iCloud'dan inmemiş. Dosyalar uygulamasında yanındaki \
            bulut simgesine dokunup indirin, sonra tekrar deneyin.
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

    /// One file to bring in, with the scoped parent it was reached through.
    private struct Source {
        let url: URL
        let name: String
        let size: Int64
    }

    /// Copies the importable files of `sources` into the tier's own directory.
    ///
    /// `sources` may be folders, individual files, or a mix; each is expected
    /// to carry its own security scope from the picker.
    static func run(
        tier: Edge0Tier,
        from sources: [URL],
        onProgress: @Sendable @escaping (_ copied: Int64, _ total: Int64) -> Void
    ) throws -> URL {
        var scoped: [URL] = []
        defer { for url in scoped { url.stopAccessingSecurityScopedResource() } }
        for url in sources where url.startAccessingSecurityScopedResource() {
            scoped.append(url)
        }

        let files = try collect(from: sources)
        guard !files.isEmpty else { throw Edge0ImportError.nothingSelected }
        guard files.contains(where: { $0.name == "config.json" }) else {
            throw Edge0ImportError.missingConfig
        }
        guard files.contains(where: { $0.name.hasSuffix(".safetensors") }) else {
            throw Edge0ImportError.noWeights
        }

        let total = files.reduce(Int64(0)) { $0 + $1.size }
        let free = Edge0Storage.freeDiskSpace
        guard free > total else {
            throw Edge0ImportError.notEnoughSpace(needed: total, free: free)
        }

        let manager = FileManager.default
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
            for file in files {
                try Task.checkCancellation()
                try copy(
                    file, to: staging.appendingPathComponent(file.name),
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

    /// Flattens the selection into the files worth copying, whether the user
    /// picked a folder or the files inside it.
    private static func collect(from sources: [URL]) throws -> [Source] {
        let manager = FileManager.default
        var files: [Source] = []
        var seen: Set<String> = []

        for url in sources {
            var isDirectory: ObjCBool = false
            guard manager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
                throw Edge0ImportError.notReadable(url.lastPathComponent)
            }

            let candidates: [URL]
            if isDirectory.boolValue {
                guard
                    let names = try? manager.contentsOfDirectory(atPath: url.path)
                else { throw Edge0ImportError.notReadable(url.lastPathComponent) }
                candidates = names.sorted().map { url.appendingPathComponent($0) }
            } else {
                candidates = [url]
            }

            for candidate in candidates {
                let name = candidate.lastPathComponent
                guard shouldImport(name), seen.insert(name).inserted else { continue }
                try ensureMaterialised(candidate)
                let values = try? candidate.resourceValues(forKeys: [.fileSizeKey])
                files.append(
                    Source(url: candidate, name: name, size: Int64(values?.fileSize ?? 0)))
            }
        }
        return files
    }

    /// A file in iCloud Drive may be a placeholder with no bytes behind it.
    /// Reading one silently yields nothing, so ask for it and say so plainly
    /// rather than importing an empty file.
    private static func ensureMaterialised(_ url: URL) throws {
        let values = try? url.resourceValues(forKeys: [
            .isUbiquitousItemKey, .ubiquitousItemDownloadingStatusKey,
        ])
        guard values?.isUbiquitousItem == true else { return }
        if values?.ubiquitousItemDownloadingStatus == .current { return }
        try? FileManager.default.startDownloadingUbiquitousItem(at: url)
        throw Edge0ImportError.notDownloaded(url.lastPathComponent)
    }

    /// Chunked so progress advances within a single multi-gigabyte shard and
    /// cancellation is noticed promptly.
    private static func copy(
        _ file: Source, to destination: URL,
        copied: inout Int64, total: Int64,
        onProgress: @Sendable (Int64, Int64) -> Void
    ) throws {
        let manager = FileManager.default
        guard let input = try? FileHandle(forReadingFrom: file.url) else {
            throw Edge0ImportError.notReadable(file.name)
        }
        defer { try? input.close() }

        try? manager.removeItem(at: destination)
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
