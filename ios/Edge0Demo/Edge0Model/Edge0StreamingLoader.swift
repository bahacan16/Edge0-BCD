// Loads the 35B tier without ever holding its experts in memory.
//
// The stock loader reads every safetensors shard into RAM before handing the
// weights to the model — ~23 GB for this checkpoint, which no phone can do.
// This path instead:
//
//   1. maps the shards read-only and copies in only the non-expert tensors
//      (~3 GB: embeddings, attention, norms, shared experts, lm_head),
//   2. quantizes and applies those,
//   3. swaps every MoE block's expert layer for a streaming one that reads
//      the selected experts out of the mapping per step.
//
// Everything else — tokenizer, processor, chat template — is assembled the
// same way `LLMModelFactory` does it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

enum Edge0StreamingLoaderError: LocalizedError {
    case noShards(URL)
    case noExpertTensors(String)

    var errorDescription: String? {
        switch self {
        case .noShards(let url): "Model ağırlıkları bulunamadı: \(url.lastPathComponent)"
        case .noExpertTensors(let path): "Expert tensörleri eşlenemedi: \(path)"
        }
    }
}

enum Edge0StreamingLoader {

    /// A tensor that belongs to a routed expert stack, i.e. one this loader
    /// deliberately leaves on disk. `shared_expert` is *not* one of these — it
    /// runs for every token and stays resident.
    static func isExpertTensor(_ name: String) -> Bool {
        name.contains(".switch_mlp.") || name.contains(".experts.")
    }

    static func load(
        tier: Edge0Tier,
        directory: URL,
        tokenizerLoader: any TokenizerLoader,
        hotSlotsPerLayer: Int,
        loraURL: URL?
    ) async throws -> (context: ModelContext, loraReport: Edge0LoRAReport?) {
        let configURL = directory.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        let baseConfiguration = try JSONDecoder().decode(
            BaseConfiguration.self, from: configData)
        let configuration = try JSONDecoder().decode(
            E0Qwen35Configuration.self, from: configData)

        let model = E0Qwen35MoEModel(configuration)

        let shards = try SafetensorsShardSet(directory: directory)
        guard !shards.tensorNames.isEmpty else {
            throw Edge0StreamingLoaderError.noShards(directory)
        }

        // 1. Resident weights only.
        var weights: [String: MLXArray] = [:]
        for name in shards.tensorNames where !isExpertTensor(name) {
            guard let shard = shards.shard(for: name), let entry = shard.entries[name],
                let dtype = SafetensorsMmap.dtype(from: entry.dtype)
            else { continue }
            weights[name] = try shard.whole(tensor: name, as: dtype)
        }
        weights = model.sanitize(weights: weights)

        // 2. Quantize exactly the layers the checkpoint ships scales for.
        let perLayer = baseConfiguration.perLayerQuantization
        let quantization = baseConfiguration.quantization
        if perLayer != nil || quantization != nil {
            quantize(model: model) { path, _ in
                guard weights["\(path).scales"] != nil else { return nil }
                if let perLayer {
                    return perLayer.quantization(layer: path)?.asTuple
                }
                return quantization?.asTuple
            }
        }

        // Expert weights are intentionally absent, so verification has to be off.
        model.update(parameters: ModuleParameters.unflattened(weights))
        weights.removeAll()

        // 3. Route every MoE block through the streaming expert pool.
        let installed = installStreamingExperts(
            in: model, shards: shards, hotSlots: hotSlotsPerLayer,
            groupSize: quantization?.groupSize ?? 64, bits: quantization?.bits ?? 4)
        guard installed > 0 else {
            throw Edge0StreamingLoaderError.noExpertTensors(directory.lastPathComponent)
        }

        eval(model)

        // 4. LoRA, tokenizer, processor — as the stock factory would.
        var report: Edge0LoRAReport?
        if let loraURL, FileManager.default.fileExists(atPath: loraURL.path) {
            report = try Edge0LoRA.apply(
                to: model, fileURL: loraURL, rank: 16, alpha: 32.0)
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

    /// Replaces each MoE block's resident expert layer with a streaming one.
    /// Returns how many blocks were converted.
    private static func installStreamingExperts(
        in model: Module, shards: SafetensorsShardSet, hotSlots: Int, groupSize: Int, bits: Int
    ) -> Int {
        var installed = 0
        for (path, module) in model.namedModules() {
            guard let block = module as? E0Qwen35SparseMoeBlock else { continue }
            let switchPath = "\(path).switch_mlp"
            guard let names = ExpertTensorNames(modulePath: switchPath, shards: shards) else {
                continue
            }
            block.switchMLP = Edge0StreamingSwitchGLU(
                shards: shards,
                names: names,
                quantization: ExpertQuantization(groupSize: groupSize, bits: bits),
                hotSlots: hotSlots
            )
            installed += 1
        }
        return installed
    }
}

/// Mirrors mlx-swift-lm's own (private) `LLMUserInputProcessor`.
struct Edge0UserInputProcessor: UserInputProcessor {
    let tokenizer: any MLXLMCommon.Tokenizer
    let messageGenerator: any MessageGenerator

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(tokens))
        } catch {
            // Most likely a checkpoint with no chat template; fall back to the
            // same plain-text join the stock processor uses.
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            return LMInput(tokens: MLXArray(tokenizer.encode(text: prompt)))
        }
    }
}
