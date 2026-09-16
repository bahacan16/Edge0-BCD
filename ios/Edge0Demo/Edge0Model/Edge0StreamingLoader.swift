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

import Darwin
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
        expertCacheBudgetBytes: Int,
        automaticMemory: Bool,
        loraURL: URL?,
        prerouterURL: URL?
    ) async throws -> (context: ModelContext, loraReport: Edge0LoRAReport?) {
        let configURL = directory.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        let baseConfiguration = try JSONDecoder().decode(
            BaseConfiguration.self, from: configData)
        let configuration = try JSONDecoder().decode(
            E0Qwen35Configuration.self, from: configData)

        // Constructing the model also constructs full-size expert layers —
        // [256, out, in] per projection per layer, ~15 GB if it were real. It
        // is not: MLX builds those initializers lazily, and step 3 below
        // replaces the whole expert layer before anything evaluates them, so
        // they never leave the graph. Nothing may call `eval` on this model
        // until the experts have been swapped.
        let model = E0Qwen35MoEModel(configuration)

        let shards = try SafetensorsShardSet(directory: directory)
        guard !shards.tensorNames.isEmpty else {
            throw Edge0StreamingLoaderError.noShards(directory)
        }

        // 1. Resident weights only.
        //
        // A tensor that cannot be read is fatal, not skippable: skipping one
        // leaves the model holding the random values its initializer produced,
        // and since expert weights are legitimately absent here the parameter
        // update cannot verify its keys and would not notice. The result would
        // load cleanly and answer nonsense.
        var weights: [String: MLXArray] = [:]
        for name in shards.tensorNames where !isExpertTensor(name) {
            guard let shard = shards.shard(for: name), let entry = shard.entries[name] else {
                throw SafetensorsError.unknownTensor(name)
            }
            guard let dtype = SafetensorsMmap.dtype(from: entry.dtype) else {
                throw SafetensorsError.unsupportedDType(tensor: name, dtype: entry.dtype)
            }
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
        //
        //    The prerouter's heads land in memory after this, so the space they
        //    will take has to be kept out of the expert cache's share now —
        //    sizing the cache against memory the prerouter is about to claim is
        //    how a load that fits becomes a load that gets the app killed.
        //
        //    Its resident size, once: installing it does briefly hold the file
        //    twice (per-head copies, then the stacks built from them), but that
        //    happens moments from here with the expert cache still empty, so
        //    reserving for the peak would take slots away for memory nothing
        //    holds by the time the cache is big enough to care.
        let prerouterBytes = prerouterURL.map(fileSize(of:)) ?? 0
        let installed = installStreamingExperts(
            in: model, tier: tier, shards: shards,
            budgetBytes: expertCacheBudgetBytes, automatic: automaticMemory,
            prerouterBytes: prerouterBytes,
            groupSize: quantization?.groupSize ?? 64, bits: quantization?.bits ?? 4)
        guard installed > 0 else {
            throw Edge0StreamingLoaderError.noExpertTensors(directory.lastPathComponent)
        }

        Edge0Log.write("akıtmalı expert katmanı kuruldu: \(installed) blok")

        // Safe now, and only now: the placeholder expert weights are gone.
        eval(model)
        Edge0Log.write("model değerlendirildi")

        // 4. LoRA, tokenizer, processor — as the stock factory would.
        var report: Edge0LoRAReport?
        if let loraURL, FileManager.default.fileExists(atPath: loraURL.path) {
            report = try Edge0LoRA.apply(
                to: model, fileURL: loraURL, rank: 16, alpha: 32.0)
        }

        // 5. The prerouter, last: it walks the settled module tree and hangs on
        //    to each MoE block, and LoRA has just replaced some of their
        //    children.
        //
        //    A failure here is reported, not thrown. The prerouter is a speed
        //    feature on top of a model that already works without it, and the
        //    likeliest failure is the adapter file simply not being on the
        //    device — refusing to load a 23 GB checkpoint over that would be
        //    the worse outcome by far.
        if let prerouterURL, FileManager.default.fileExists(atPath: prerouterURL.path) {
            do {
                let heads = try Edge0Prerouter.install(
                    into: model, fileURL: prerouterURL,
                    startLayer: tier.prerouterStartLayer, hidden: tier.prerouterHiddenSize)
                Edge0Log.write(
                    "prerouter: \(heads) baş, \(tier.prerouterStartLayer). katmandan itibaren")
            } catch {
                Edge0Log.failure("prerouter kurulamadı", error)
            }
        } else if prerouterURL != nil {
            Edge0Log.write(
                "prerouter dosyası yok: \(tier.prerouterFileName ?? "-") — kapılarla çalışılıyor")
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

    /// Bytes on disk, or zero for a file that is not there.
    private static func fileSize(of url: URL) -> Int {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes?[.size] as? NSNumber)?.intValue ?? 0
    }

    /// Replaces each MoE block's resident expert layer with a streaming one.
    /// Returns how many blocks were converted.
    ///
    /// The cache is sized in bytes rather than in experts. The budget is what
    /// the user actually cares about, and it has to be split across every MoE
    /// layer: forty layers each holding sixteen experts is not "sixteen
    /// experts", it is forty times that, and on this checkpoint it is well over
    /// a gigabyte.
    private static func installStreamingExperts(
        in model: Module, tier: Edge0Tier, shards: SafetensorsShardSet,
        budgetBytes: Int, automatic: Bool, prerouterBytes: Int,
        groupSize: Int, bits: Int
    ) -> Int {
        var blocks: [(block: E0Qwen35SparseMoeBlock, names: ExpertTensorNames)] = []
        for (path, module) in model.namedModules() {
            guard let block = module as? E0Qwen35SparseMoeBlock else { continue }
            guard
                let names = ExpertTensorNames(
                    modulePath: "\(path).switch_mlp", shards: shards)
            else { continue }
            blocks.append((block, names))
        }
        guard let first = blocks.first else { return 0 }

        // One expert of THIS checkpoint, not of a checkpoint in general: a 35B
        // expert and an 8B one are not the same size, and the slot count is the
        // number that decides decode speed. Every miss is a read from storage,
        // and with 256 experts a small cache means nearly every layer of every
        // token goes to disk.
        //
        // The prerouter's heads are about to land in memory too, so their space
        // is taken out of what the experts may have before the plan is made.
        let perExpert = first.names.bytesPerExpert(shards: shards)
        let plan = Edge0MemoryPlanner.make(
            tier: tier, automatic: automatic,
            manualBudgetBytes: budgetBytes,
            perExpertBytes: perExpert, layerCount: blocks.count,
            alsoReserving: prerouterBytes)
        let slots = plan.slotsPerLayer

        Edge0Log.write(
            "expert önbelleği: \(plan.detail) → katman başına \(slots) slot"
                + " (\(blocks.count) katman, expert başına \(perExpert / 1024) KB,"
                + " toplam \(plan.budgetBytes / 1_048_576) MB)")

        // How many distinct experts one call may stack before it starts
        // splitting the prompt into pieces. Bytes, not slots: the stack is a
        // transient copy freed when the layer finishes, and every split makes
        // the pieces re-read what the whole would have read once.
        let maxStack = max(16, 320 * 1024 * 1024 / max(1, perExpert))

        for (block, names) in blocks {
            let streaming = Edge0StreamingSwitchGLU(
                shards: shards,
                names: names,
                quantization: ExpertQuantization(groupSize: groupSize, bits: bits),
                hotSlots: slots,
                maxExpertsPerCall: maxStack
            )
            // NOT `block.switchMLP = streaming`. @ModuleInfo's setter traps
            // outright once the property holds a value — assigning through it
            // would leave Module's own child cache pointing at the layer that
            // was replaced. `update(modules:)` is the supported route, and the
            // key is the one @ModuleInfo was declared with.
            block.update(modules: .unflattened([("switch_mlp", streaming)]))
        }
        return blocks.count
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
