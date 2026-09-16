// Loads a tier whose experts fit in memory, without going through
// mlx-swift-lm's factory.
//
// The factory reads `model_type` out of config.json and refuses the file
// without it. edge0's 8B checkpoint does not have one — it identifies itself
// the way a `trust_remote_code` model does, through `architectures` and
// `auto_map` pointing at its own Python classes — so every attempt to load that
// tier died at the first line with "Missing field 'model_type'". The field is
// not missing by mistake and it is not the user's to add; a loader that already
// knows which architecture it is building has no business asking the file to
// name it.
//
// So this does what the streaming loader does, minus the streaming: maps the
// shards, copies every tensor in, quantizes exactly the layers the checkpoint
// ships scales for, and assembles the same tokenizer and processor around it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

enum Edge0ResidentLoader {

    /// The quantization block, read directly rather than through
    /// `BaseConfiguration` — which is the type that wants `model_type`.
    private struct Quantization: Codable {
        let groupSize: Int
        let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    private struct ConfigurationHead: Codable {
        let quantization: Quantization?
    }

    static func load(
        tier: Edge0Tier,
        directory: URL,
        tokenizerLoader: any TokenizerLoader,
        loraURL: URL?
    ) async throws -> (context: ModelContext, loraReport: Edge0LoRAReport?) {
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let head = try JSONDecoder().decode(ConfigurationHead.self, from: configData)
        let configuration = try JSONDecoder().decode(
            Edge0BailingConfiguration.self, from: configData)
        let model = Edge0BailingModel(configuration)

        let shards = try SafetensorsShardSet(directory: directory)
        guard !shards.tensorNames.isEmpty else {
            throw Edge0StreamingLoaderError.noShards(directory)
        }

        // Everything, experts included: this tier is small enough to stay
        // resident, which is the whole reason it is not on the streaming path.
        // A tensor that cannot be read is fatal rather than skippable — the
        // model would otherwise keep whatever its initializer produced and load
        // cleanly into nonsense.
        var weights: [String: MLXArray] = [:]
        for name in shards.tensorNames {
            guard let shard = shards.shard(for: name), let entry = shard.entries[name] else {
                throw SafetensorsError.unknownTensor(name)
            }
            guard let dtype = SafetensorsMmap.dtype(from: entry.dtype) else {
                throw SafetensorsError.unsupportedDType(tensor: name, dtype: entry.dtype)
            }
            weights[name] = try shard.whole(tensor: name, as: dtype)
        }
        weights = model.sanitize(weights: weights)

        if let quantization = head.quantization {
            quantize(model: model) { path, _ in
                guard weights["\(path).scales"] != nil else { return nil }
                return (groupSize: quantization.groupSize, bits: quantization.bits)
            }
        }

        model.update(parameters: ModuleParameters.unflattened(weights))
        weights.removeAll()
        eval(model)
        Edge0Log.write("model değerlendirildi (yerleşik)")

        var report: Edge0LoRAReport?
        if let loraURL, FileManager.default.fileExists(atPath: loraURL.path) {
            report = try Edge0LoRA.apply(to: model, fileURL: loraURL, rank: 16, alpha: 32.0)
        }

        let tokenizer = try await tokenizerLoader.load(from: directory)
        let modelConfiguration = ModelConfiguration(
            directory: directory, eosTokenIds: tier.eosTokenIds)
        let processor = Edge0UserInputProcessor(
            tokenizer: tokenizer,
            messageGenerator: model.messageGenerator(tokenizer: tokenizer))

        let context = ModelContext(
            configuration: modelConfiguration, model: model, processor: processor,
            tokenizer: tokenizer)
        return (context, report)
    }
}
