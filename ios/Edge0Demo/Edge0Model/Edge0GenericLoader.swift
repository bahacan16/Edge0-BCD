// Loads a checkpoint this app was not written for.
//
// The two edge0 tiers each have a loader that already knows what it is
// building. This one has to ask, and the only thing it can ask is config.json:
// `model_type` names the architecture, `quantization` names the format, and
// `text_config` — when there is one — holds the language model's shape while
// the top level describes a wrapper the app does not care about.
//
// Only architectures this app can actually build are accepted, and the refusal
// names the type it found. Loading a checkpoint through the wrong port does
// not fail loudly: the tensors land in shapes that happen to fit, the health
// check passes, and the model answers fluent nonsense. A model that will not
// load is a much better outcome than one that lies.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

enum Edge0GenericLoaderError: LocalizedError {
    case unsupportedArchitecture(String, [String])
    case noConfiguration(URL)
    case streamingRequired(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedArchitecture(let type, let architectures):
            let named = type.isEmpty ? architectures.joined(separator: ", ") : type
            return """
                Bu mimari desteklenmiyor: \(named.isEmpty ? "bilinmiyor" : named). \
                Uygulama şimdilik qwen3_5 ailesini yükleyebiliyor. Modelin \
                config.json'undaki model_type değeri budur.
                """
        case .noConfiguration(let url):
            return "config.json okunamadı: \(url.lastPathComponent)"
        case .streamingRequired(let name):
            return """
                \(name) uzman ağırlıklarını diskten akıtmayı gerektiriyor; bu yol \
                yalnızca edge0'ın 35B checkpoint'i için yazıldı. Belleğe sığan bir \
                model deneyin.
                """
        }
    }
}

enum Edge0GenericLoader {

    /// The quantization block, read on its own. `BaseConfiguration` would do
    /// this too, but it insists on `model_type` being present and spelled the
    /// way it expects, which is exactly the assumption this loader cannot make.
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

    /// Architectures this loader can build, by `model_type`.
    ///
    /// `qwen3_5` is the dense sibling of the 35B tier's `qwen3_5_moe`: same
    /// hybrid backbone, same gated-delta linear attention on three layers out
    /// of four, same partial rotary and mrope — the vendored port already
    /// branches to a plain MLP when the config declares no experts, and its
    /// top-level wrapper already drops a vision tower's weights and remaps
    /// `model.language_model.*`. So this costs nothing but the routing.
    private static let supported: Set<String> = ["qwen3_5", "qwen3_5_moe"]

    static func load(
        tier: Edge0Tier,
        directory: URL,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext {
        let configURL = directory.appending(component: "config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw Edge0GenericLoaderError.noConfiguration(configURL)
        }
        guard let info = Edge0CustomModelRegistry.read(directory: directory) else {
            throw Edge0GenericLoaderError.noConfiguration(configURL)
        }
        guard supported.contains(info.modelType) else {
            throw Edge0GenericLoaderError.unsupportedArchitecture(
                info.modelType, info.architectures)
        }
        // A MoE checkpoint big enough to need streaming is not this loader's
        // to attempt: it would map every expert into memory and be killed.
        if info.isMoE, info.byteCount > 12 * 1_073_741_824 {
            throw Edge0GenericLoaderError.streamingRequired(info.name)
        }

        Edge0Log.write(
            "genel yükleyici: \(info.modelType) · \(info.layerCount) katman"
                + " · sözlük \(info.vocabularySize)"
                + " · bağlam \(info.contextLength)"
                + (info.isQuantized ? " · \(info.quantBits)-bit/\(info.quantGroupSize)" : ""))

        let head = try JSONDecoder().decode(ConfigurationHead.self, from: configData)
        let configuration = try JSONDecoder().decode(
            E0Qwen35Configuration.self, from: configData)
        let model = E0Qwen35Model(configuration)

        let shards = try SafetensorsShardSet(directory: directory)
        guard !shards.tensorNames.isEmpty else {
            throw Edge0StreamingLoaderError.noShards(directory)
        }

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
        // This is where a vision tower's tensors are dropped and the language
        // model's are moved under the wrapper's key.
        weights = model.sanitize(weights: weights)

        if let quantization = head.quantization {
            quantize(model: model) { path, _ in
                // Only the layers the checkpoint actually shipped scales for:
                // a quantized export leaves some modules in full precision and
                // quantizing those too would look for tensors that do not
                // exist.
                guard weights["\(path).scales"] != nil else { return nil }
                return (groupSize: quantization.groupSize, bits: quantization.bits)
            }
        }

        model.update(parameters: ModuleParameters.unflattened(weights))
        weights.removeAll()
        eval(model)
        Edge0Log.write("model değerlendirildi (genel)")

        let tokenizer = try await tokenizerLoader.load(from: directory)
        // The eos ids come from the checkpoint rather than from a table: this
        // is someone else's model and its end-of-turn token is its own.
        let modelConfiguration = ModelConfiguration(
            directory: directory, eosTokenIds: Set(info.eosTokenIds))
        let processor = Edge0UserInputProcessor(
            tokenizer: tokenizer,
            messageGenerator: model.messageGenerator(tokenizer: tokenizer))

        return ModelContext(
            configuration: modelConfiguration, model: model, processor: processor,
            tokenizer: tokenizer)
    }
}
