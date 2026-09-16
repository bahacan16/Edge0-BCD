// edge0's trained prerouter — a port of edge0/src/edge0/prerouter (Edge0-AI/edge0,
// Apache-2.0) for the 35B tier's qwen backbone.
//
// Why this exists, in one sentence: without it the streaming tier learns which
// experts it needs at the exact moment it needs them, so every layer of every
// token is a fresh trip to storage, forty of them in a chain, and no amount of
// concurrency inside a single layer can hide that.
//
// The prerouter breaks the chain. Each layer N carries a small trained head
// that reads layer N's own MoE input and predicts layer N+1's routing. Every
// head runs once at the step boundary, as one stacked batch, and the result is
// the whole next token's expert set for thirty-three layers at once — known
// before the forward that needs it has started. Two things follow:
//
//   * the reads for the entire next token can be issued in parallel, in the
//     background, while the current step is still finishing;
//   * the prediction IS the routing (the block consumes `pred_inds` instead of
//     calling its gate), so what was prefetched is exactly what gets used.
//
// The second point is not an optimisation, it is what makes the first one
// sound, and it is why edge0 ships a Recover-LoRA alongside: the adapters were
// trained with the prerouter in the loop. Running the LoRA without the
// prerouter — which is what this app did until now — is the configuration
// neither half was trained for.
//
// Feature vector per head: concat[hidden, this-token top-k one-hot,
// prev-token top-k one-hot]; head = fc1 -> exact(erf) GELU -> fc2, plus a
// `linear_init` skip taken on the same features.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Edge0PrerouterError: LocalizedError {
    case noHeads(URL)
    case missingOwners([Int])
    case shapeMismatch(String)

    var errorDescription: String? {
        switch self {
        case .noHeads(let url):
            "Prerouter başlıkları okunamadı: \(url.lastPathComponent)"
        case .missingOwners(let layers):
            "Prerouter ağırlıklarında eksik katmanlar: "
                + layers.prefix(5).map { "\($0)" }.joined(separator: ", ")
        case .shapeMismatch(let detail):
            "Prerouter ağırlık boyutu uyuşmuyor: \(detail)"
        }
    }
}

/// One layer's prediction for the next decode step.
struct Edge0Routing {
    let indices: MLXArray
    let scores: MLXArray
}

final class Edge0Prerouter {

    let numExperts: Int
    let topK: Int
    /// First *consuming* layer. Layer `startLayer - 1` is the first owner, and
    /// it has nothing to consume itself.
    let startLayer: Int
    let owners: [Int]

    /// Stacked head weights, one slice per owner, in `owners` order.
    private let w1: MLXArray  // [n, featureCount, hidden]
    private let w2: MLXArray  // [n, hidden, numExperts]
    private let wl: MLXArray  // [n, featureCount, numExperts]
    private let dtype: DType

    /// `[0, 1, ... numExperts-1]`, for building the top-k one-hot features by
    /// broadcast comparison; built once.
    private let expertIds: MLXArray
    /// The all-zeros one-hot handed to the "previous token" feature on the
    /// first decode step of a turn, when there is no previous token.
    private let zeroOneHot: MLXArray

    // Decode state. Keyed by layer index.
    private var capturedInput: [Int: MLXArray] = [:]
    private var oneHots: [Int: MLXArray] = [:]
    private var previousOneHots: [Int: MLXArray] = [:]
    private var predictions: [Int: Edge0Routing] = [:]

    /// The streaming expert layer of each *consuming* layer, so a prediction
    /// can be turned into page-ins straight away.
    private var streaming: [Int: Edge0StreamingSwitchGLU] = [:]

    /// Serial on purpose: one step's prefetch should finish before the next
    /// one's is issued, and the fan-out happens inside each job anyway.
    ///
    /// Deliberately BELOW the generation thread's priority. Every page-in is a
    /// separate small read — the mapping is advised MADV_RANDOM, so the kernel
    /// does one page per fault and no readahead — and there are only so many
    /// threads to make them on. A prefetch running at the same priority as the
    /// forward is not just working alongside it, it is bidding for the same
    /// threads with reads the model does not need for another thirty layers,
    /// and the layer that needs its experts *now* waits behind them. Speculative
    /// work must lose that race, always.
    private let prefetchQueue = DispatchQueue(
        label: "ai.edge0.prerouter.prefetch", qos: .utility)

    private var _steps = 0
    private let counterLock = NSLock()
    /// How many decode steps were routed by a prediction rather than by a gate.
    var predictedSteps: Int { counterLock.withLock { _steps } }

    private init(
        numExperts: Int, topK: Int, startLayer: Int, owners: [Int],
        w1: MLXArray, w2: MLXArray, wl: MLXArray, dtype: DType
    ) {
        self.numExperts = numExperts
        self.topK = topK
        self.startLayer = startLayer
        self.owners = owners
        self.w1 = w1
        self.w2 = w2
        self.wl = wl
        self.dtype = dtype
        self.expertIds = MLXArray((0 ..< numExperts).map { Int32($0) })
        self.zeroOneHot = MLXArray(
            Array(repeating: Float(0), count: numExperts)
        ).reshaped([1, 1, numExperts])
    }

    // MARK: - Install

    /// Reads `fileURL`'s heads and wires them into `model`'s MoE blocks.
    ///
    /// Returns the number of heads installed, and throws rather than installing
    /// a partial one: a missing head would leave that layer on its gate while
    /// its neighbours route from predictions, which is neither of the two
    /// configurations anything was trained for, and not something to discover
    /// later as an unexplained drop in answer quality. The caller decides what
    /// to do about the throw — here, load without a prerouter and say so.
    @discardableResult
    static func install(
        into model: E0Qwen35Model, fileURL: URL, startLayer: Int, hidden: Int
    ) throws -> Int {
        let text = model.languageModel
        let configuration = text.configuration
        let layerCount = configuration.hiddenLayers
        let numExperts = configuration.numExperts
        let topK = configuration.numExpertsPerTok
        let featureCount = configuration.hiddenSize + 2 * numExperts
        let owners = Array((startLayer - 1) ..< (layerCount - 1))
        guard !owners.isEmpty else { throw Edge0PrerouterError.noHeads(fileURL) }

        var raw = try loadArrays(url: fileURL)
        var heads: [Int: [String: MLXArray]] = [:]
        for (key, value) in raw {
            let parts = key.split(separator: ".", omittingEmptySubsequences: false)
            guard parts.count >= 4, parts[0] == "layers", let owner = Int(parts[1]),
                parts[parts.count - 1] == "weight"
            else { continue }
            let part = String(parts[parts.count - 2])
            guard part == "fc1" || part == "fc2" || part == "linear_init" else { continue }
            heads[owner, default: [:]][part] = value
        }
        guard !heads.isEmpty else { throw Edge0PrerouterError.noHeads(fileURL) }

        let missing = owners.filter { heads[$0]?.count != 3 }
        guard missing.isEmpty else { throw Edge0PrerouterError.missingOwners(missing) }

        let dtype: DType = .float16
        func stack(_ part: String, rows: Int, columns: Int) throws -> MLXArray {
            var slices: [MLXArray] = []
            slices.reserveCapacity(owners.count)
            for owner in owners {
                let weight = heads[owner]![part]!
                guard weight.ndim == 2, weight.dim(0) == rows, weight.dim(1) == columns else {
                    throw Edge0PrerouterError.shapeMismatch(
                        "layers.\(owner).\(part).weight \(weight.shape) != [\(rows), \(columns)]")
                }
                slices.append(weight.asType(dtype))
            }
            // Stored as torch `[out, in]`; the batched matmul below wants
            // `[in, out]` per head.
            return MLX.stacked(slices).transposed(0, 2, 1)
        }

        let w1 = try stack("fc1", rows: hidden, columns: featureCount)
        let w2 = try stack("fc2", rows: numExperts, columns: hidden)
        let wl = try stack("linear_init", rows: numExperts, columns: featureCount)
        // Materialise the stacks, then drop the per-head copies they were built
        // from: until the eval they are still referenced by the lazy graph, and
        // after it they are ~140 MB of nothing on a phone that has about two
        // gigabytes left at this point.
        eval(w1, w2, wl)
        heads.removeAll()
        raw.removeAll()

        let prerouter = Edge0Prerouter(
            numExperts: numExperts, topK: topK, startLayer: startLayer, owners: owners,
            w1: w1, w2: w2, wl: wl, dtype: dtype)

        // The owners are exactly the blocks that have to capture features, so
        // they are the blocks the prerouter drives. Note the two ends: layer
        // `startLayer - 1` owns a head but has none pointed at it, and the last
        // layer is not an owner at all, so both keep their own gate. That
        // leaves layer `layerCount - 2`'s prediction unused, which is what
        // upstream does too — one head's worth of arithmetic, and matching the
        // reference implementation is worth more than saving it.
        var installed = 0
        for (path, module) in model.namedModules() {
            guard let block = module as? E0Qwen35SparseMoeBlock,
                let key = LayerScopedName(path: path), key.relativePath == "mlp",
                owners.contains(key.layerIndex)
            else { continue }
            block.prerouter = prerouter
            block.prerouterLayer = key.layerIndex
            installed += 1
        }
        guard installed == owners.count else {
            throw Edge0PrerouterError.shapeMismatch(
                "\(installed) MoE bloğu bulundu, \(owners.count) bekleniyordu")
        }

        // Consumers, for the prefetch. The head owned by layer N predicts for
        // layer N+1, so that is the layer whose experts the prediction pages in
        // — but only where the prediction is also what that layer routes with.
        // The last layer is handed a prediction it never reads, and prefetching
        // for it would be reads spent on experts chosen by a different function
        // than the one about to pick them.
        for (path, module) in model.namedModules() {
            guard let block = module as? E0Qwen35SparseMoeBlock,
                let key = LayerScopedName(path: path), key.relativePath == "mlp",
                let experts = block.switchMLP as? Edge0StreamingSwitchGLU,
                owners.contains(key.layerIndex), owners.contains(key.layerIndex - 1)
            else { continue }
            prerouter.streaming[key.layerIndex] = experts
        }

        text.prerouter = prerouter
        return installed
    }

    // MARK: - Block callbacks

    /// The prediction this layer should route with, or nil when there is none
    /// yet (the first decode step of a turn, and every layer below
    /// `startLayer`) and the block should fall back to its own gate.
    func prediction(for layer: Int) -> Edge0Routing? {
        guard layer >= startLayer else { return nil }
        return predictions[layer]
    }

    /// `[B, T, k]` expert indices -> `[B, T, numExperts]` one-hot, by
    /// comparing each slot against every expert id and summing over the slots.
    func oneHot(_ indices: MLXArray) -> MLXArray {
        // `argPartition` hands back unsigned indices; matching the id vector's
        // type keeps the comparison from going through a promotion no one asked
        // for. Expert ids are small, so the cast cannot lose anything.
        let expanded = MLX.expandedDimensions(indices.asType(.int32), axis: -1)
        return (expanded .== expertIds).asType(.float32).sum(axis: -2)
    }

    /// Records what this layer's head will need at the step boundary.
    func capture(layer: Int, input: MLXArray, oneHot: MLXArray) {
        capturedInput[layer] = input
        oneHots[layer] = oneHot
    }

    // MARK: - Step boundary

    /// Drops every decode-time capture. Called on a multi-token forward, i.e.
    /// at the start of each prompt: otherwise the first head batch of a new
    /// turn would run on the previous turn's last hidden state and hand the
    /// first decode step a prediction belonging to another conversation.
    func reset() {
        capturedInput.removeAll()
        oneHots.removeAll()
        previousOneHots.removeAll()
        predictions.removeAll()
    }

    /// Runs every head as one batch and publishes the next step's routing.
    ///
    /// `logits` is the forward's own output, evaluated here for the same reason
    /// edge0 evaluates it: the captured features are lazy until something
    /// forces them, and the expert indices have to reach the CPU before the
    /// page-ins can be issued.
    func stageAll(logits: MLXArray) {
        Edge0Meter.measure(Edge0Meter.addStageTime) { stage(logits: logits) }
    }

    private func stage(logits: MLXArray) {
        defer { advance() }

        var inputs: [MLXArray] = []
        var current: [MLXArray] = []
        var previous: [MLXArray] = []
        inputs.reserveCapacity(owners.count)
        for owner in owners {
            guard let input = capturedInput[owner], let oneHot = oneHots[owner] else { return }
            inputs.append(input)
            current.append(oneHot)
            previous.append(previousOneHots[owner] ?? zeroOneHot)
        }
        guard inputs.count == owners.count else { return }

        eval(logits)

        let count = owners.count
        var features = MLX.concatenated(
            [MLX.stacked(inputs), MLX.stacked(current), MLX.stacked(previous)], axis: -1)
        features = features.reshaped([count, 1, -1])
        if features.dtype != dtype { features = features.asType(dtype) }

        let hidden = MLXNN.gelu(MLX.matmul(features, w1))
        let headLogits = MLX.matmul(features, wl) + MLX.matmul(hidden, w2)

        // The qwen router's own math: precise softmax, top-k, renormalize.
        let gates = MLX.softmax(headLogits.asType(.float32), axis: -1, precise: true)
        let kth = numExperts - topK
        let indices = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
        var scores = MLX.takeAlong(gates, indices, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)
        eval(indices, scores)

        let flat = indices.asArray(Int32.self)
        var ranges: [ExpertByteRange] = []
        for (position, owner) in owners.enumerated() {
            let consumer = owner + 1
            predictions[consumer] = Edge0Routing(
                indices: indices[position].reshaped([1, 1, topK]),
                scores: scores[position].reshaped([1, 1, topK]))

            guard let experts = streaming[consumer] else { continue }
            let selected = Array(
                Set(flat[(position * topK) ..< ((position + 1) * topK)])).sorted()
            ranges.append(contentsOf: experts.coldRanges(for: selected))
        }
        counterLock.withLock { _steps += 1 }

        // The whole next token's cold reads, issued at once and off this
        // thread. This is the point of the exercise: the storage gets a
        // hundred-odd independent requests to work on while the current step
        // finishes, instead of four at a time forty times in a row.
        guard !ranges.isEmpty else { return }
        prefetchQueue.async {
            let start = CFAbsoluteTimeGetCurrent()
            Edge0ExpertPaging.fault(ranges)
            Edge0Meter.addPrefetch(
                seconds: CFAbsoluteTimeGetCurrent() - start, ranges: ranges.count)
        }
    }

    /// Rolls this step's one-hots into the "previous token" feature.
    private func advance() {
        previousOneHots = oneHots
        oneHots.removeAll()
        capturedInput.removeAll()
    }
}
