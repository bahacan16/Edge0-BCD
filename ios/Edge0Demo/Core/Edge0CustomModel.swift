// A model the user brought themselves.
//
// The two edge0 tiers are known quantities: their layer counts, expert counts,
// eos ids and sampling defaults are compiled in because they come from the
// upstream repo and cannot change under us. A checkpoint dropped into
// `Models/custom` is the opposite — nothing about it is known until its
// config.json is read, and it can be replaced between launches.
//
// So `Edge0Tier.custom` is a slot rather than a description, and this is what
// fills it. Every property the enum answers for that case comes from here, and
// when the slot is empty the answers are the neutral ones that make the card
// read as "bring a model" instead of as a broken edge0 tier.

import Foundation

/// What a config.json says about a checkpoint, in the terms this app needs.
struct Edge0CustomModelInfo: Codable, Sendable, Equatable {
    var name: String
    /// `model_type` at the top level: what decides which port loads it.
    var modelType: String
    var architectures: [String]
    var layerCount: Int
    var hiddenSize: Int
    var vocabularySize: Int
    var contextLength: Int
    var quantBits: Int
    var quantGroupSize: Int
    var expertCount: Int
    var expertsPerToken: Int
    var eosTokenIds: [Int]
    var byteCount: Int64

    var isQuantized: Bool { quantBits > 0 }
    var isMoE: Bool { expertCount > 0 }

    /// One line under the name on the models card.
    var tagline: String {
        var pieces = [modelType]
        pieces.append("\(layerCount) katman")
        if isMoE { pieces.append("\(expertCount) expert") }
        if isQuantized { pieces.append("\(quantBits)-bit") }
        return pieces.joined(separator: " · ")
    }
}

/// The single custom slot, read from disk and remembered between launches.
///
/// Deliberately not main-actor: `Edge0Tier`'s properties are read from the
/// loaders, which run off the main thread, and a model's layer count is not
/// something to hop actors for.
enum Edge0CustomModelRegistry {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cached: Edge0CustomModelInfo?
    nonisolated(unsafe) private static var loaded = false

    private static let defaultsKey = "edge0.customModel"

    static var current: Edge0CustomModelInfo? {
        lock.withLock {
            if !loaded {
                loaded = true
                if let data = UserDefaults.standard.data(forKey: defaultsKey) {
                    cached = try? JSONDecoder().decode(Edge0CustomModelInfo.self, from: data)
                }
            }
            return cached
        }
    }

    private static func store(_ info: Edge0CustomModelInfo?) {
        lock.withLock {
            loaded = true
            cached = info
            if let info, let data = try? JSONEncoder().encode(info) {
                UserDefaults.standard.set(data, forKey: defaultsKey)
            } else {
                UserDefaults.standard.removeObject(forKey: defaultsKey)
            }
        }
    }

    /// Reads the folder and remembers what it found. Returns nil — and forgets
    /// whatever was there — when the folder no longer holds a checkpoint, so a
    /// deleted model does not keep describing itself on the card.
    @discardableResult
    static func rescan(directory: URL?) -> Edge0CustomModelInfo? {
        guard let directory, let info = read(directory: directory) else {
            if current != nil { store(nil) }
            return nil
        }
        if info != current { store(info) }
        return info
    }

    static func clear() { store(nil) }

    /// Parses a config.json into what the app needs from it.
    ///
    /// Multimodal checkpoints nest the language model's shape under
    /// `text_config` and leave the top level for the wrapper — so every field
    /// is looked for in the nested block first and the flat one second, which
    /// reads both layouts without having to know which one this is.
    static func read(directory: URL) -> Edge0CustomModelInfo? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        let text = (root["text_config"] as? [String: Any]) ?? root
        func integer(_ key: String, _ fallback: Int = 0) -> Int {
            (text[key] as? Int) ?? (root[key] as? Int) ?? fallback
        }

        let quantization =
            (root["quantization"] as? [String: Any])
            ?? (root["quantization_config"] as? [String: Any])
            ?? [:]

        var eos: Set<Int> = []
        for source in [root, text] {
            switch source["eos_token_id"] {
            case let single as Int: eos.insert(single)
            case let many as [Int]: eos.formUnion(many)
            default: break
            }
        }

        return Edge0CustomModelInfo(
            name: displayName(for: directory, config: root),
            modelType: (root["model_type"] as? String) ?? (text["model_type"] as? String) ?? "",
            architectures: (root["architectures"] as? [String]) ?? [],
            layerCount: integer("num_hidden_layers"),
            hiddenSize: integer("hidden_size"),
            vocabularySize: integer("vocab_size"),
            contextLength: integer("max_position_embeddings"),
            quantBits: (quantization["bits"] as? Int) ?? 0,
            quantGroupSize: (quantization["group_size"] as? Int) ?? 0,
            expertCount: integer("num_experts"),
            expertsPerToken: integer("num_experts_per_tok"),
            eosTokenIds: eos.sorted(),
            byteCount: weightBytes(in: directory))
    }

    /// The folder's own name where that says something, the checkpoint's
    /// `_name_or_path` otherwise. Dropping the downloaded folder in whole is
    /// the common case and gives the better name of the two.
    private static func displayName(for directory: URL, config: [String: Any]) -> String {
        let folder = directory.lastPathComponent
        if folder != Edge0Tier.custom.rawValue, !folder.isEmpty { return folder }
        if let path = config["_name_or_path"] as? String,
            let last = path.split(separator: "/").last
        {
            return String(last)
        }
        return "Özel model"
    }

    private static func weightBytes(in directory: URL) -> Int64 {
        let entries =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return entries.reduce(into: Int64(0)) { total, url in
            guard url.pathExtension == "safetensors" else { return }
            total += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
    }
}
