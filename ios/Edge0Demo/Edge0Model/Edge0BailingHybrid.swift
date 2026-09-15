// Port of edge0/src/edge0/backends/mlx/_impl/bailing_hybrid.py
// (Edge0-AI/edge0, Apache-2.0) — Ling 3.0's KDA (linear attention) + MLA
// (DeepSeek-style latent attention) + sigmoid group-limited-topk MoE
// hybrid backbone, used by the edge0-8b tier. The prerouter (a purely
// SSD-streaming prefetch optimization, mathematically a no-op on the
// routing itself) is intentionally not ported: this app keeps the whole
// model resident in RAM, so there is nothing to prefetch ahead of.
//
// Structural reference for MLX Swift idioms: mlx-swift-lm's
// Qwen3Next.swift (hybrid linear-attention/MoE layer scheduling) and
// DeepseekV3.swift (MLA attention + interleaved RoPE).

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Short causal depthwise conv (KDA's q/k/v pre-conv)

final class Edge0ShortConv1d: Module {
    let kernelSize: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        _conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            stride: 1, padding: 0, dilation: 1, groups: channels, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let B = x.dim(0)
        let C = x.dim(2)
        let st = state ?? MLXArray.zeros([B, kernelSize - 1, C], dtype: x.dtype)
        let convInput = concatenated([st, x], axis: 1)
        let out = silu(conv(convInput))
        let newState: MLXArray
        if kernelSize == 1 {
            newState = convInput[0..., 0 ..< 0, 0...]
        } else {
            let start = convInput.dim(1) - (kernelSize - 1)
            newState = convInput[0..., start..., 0...]
        }
        return (out, newState)
    }
}

// MARK: - KDA per-channel log-decay gate (edge0's "safe gate" law)

/// `g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))` (the safe-gate
/// branch edge0 ships); returns the LOG-domain decay — callers exponentiate
/// before feeding it to the delta-rule recurrence.
func edge0KdaGateLog(
    f: MLXArray, aLog: MLXArray, dtBias: MLXArray, safeGate: Bool, lowerBound: Float
) -> MLXArray {
    let fb = f.asType(.float32) + dtBias.asType(.float32)
    let a = exp(aLog.asType(.float32))
    if safeGate {
        return lowerBound * sigmoid(expandedDimensions(a, axis: -1) * fb)
    }
    return -expandedDimensions(a, axis: -1) * softplus(fb)
}

// MARK: - KDA (Kimi Delta Attention)

final class Edge0BailingKDA: Module {
    let numHeads: Int
    let headDim: Int
    let projDim: Int
    let safeGate: Bool
    let lowerBound: Float
    let noKdaLora: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_conv1d") var qConv1d: Edge0ShortConv1d
    @ModuleInfo(key: "k_conv1d") var kConv1d: Edge0ShortConv1d
    @ModuleInfo(key: "v_conv1d") var vConv1d: Edge0ShortConv1d

    @ModuleInfo(key: "f_proj") var fProj: Linear?
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear?
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear?
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear?
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear?

    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ args: Edge0BailingConfiguration) {
        numHeads = args.attentionHeads
        headDim = args.headDim
        projDim = numHeads * headDim
        safeGate = args.kdaSafeGate
        lowerBound = args.kdaLowerBound
        noKdaLora = args.noKdaLora
        scale = pow(Float(headDim), -0.5)

        let hidden = args.hiddenSize
        _qProj.wrappedValue = Linear(hidden, projDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, projDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, projDim, bias: false)
        _qConv1d.wrappedValue = Edge0ShortConv1d(
            channels: projDim, kernelSize: args.shortConvKernelSize)
        _kConv1d.wrappedValue = Edge0ShortConv1d(
            channels: projDim, kernelSize: args.shortConvKernelSize)
        _vConv1d.wrappedValue = Edge0ShortConv1d(
            channels: projDim, kernelSize: args.shortConvKernelSize)

        if noKdaLora {
            _fProj.wrappedValue = Linear(hidden, projDim, bias: false)
            _gProj.wrappedValue = Linear(hidden, projDim, bias: false)
        } else {
            _fAProj.wrappedValue = Linear(hidden, headDim, bias: false)
            _fBProj.wrappedValue = Linear(headDim, projDim, bias: false)
            _gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
            _gBProj.wrappedValue = Linear(headDim, projDim, bias: false)
        }

        _bProj.wrappedValue = Linear(hidden, numHeads, bias: false)
        _aLog.wrappedValue = MLXArray.zeros([numHeads])
        _dtBias.wrappedValue = MLXArray.zeros([projDim])
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _oProj.wrappedValue = Linear(projDim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let dtype = x.dtype

        let (qConvOut, newQState) = qConv1d(qProj(x), state: cache?[0])
        let (kConvOut, newKState) = kConv1d(kProj(x), state: cache?[1])
        let (vConvOut, newVState) = vConv1d(vProj(x), state: cache?[2])
        if let cache {
            cache[0] = newQState
            cache[1] = newKState
            cache[2] = newVState
        }

        var q = qConvOut.reshaped(B, T, numHeads, headDim)
        var k = kConvOut.reshaped(B, T, numHeads, headDim)
        let v = vConvOut.reshaped(B, T, numHeads, headDim)

        let qf = q.asType(.float32)
        let kf = k.asType(.float32)
        q = scale * qf / (MLX.norm(qf, axis: -1, keepDims: true) + 1e-6)
        k = kf / (MLX.norm(kf, axis: -1, keepDims: true) + 1e-6)

        var f: MLXArray
        var gate: MLXArray
        if noKdaLora {
            f = fProj!(x)
            gate = gProj!(x)
        } else {
            f = fBProj!(fAProj!(x))
            gate = gBProj!(gAProj!(x))
        }
        f = f.reshaped(B, T, numHeads, headDim)
        let gLog = edge0KdaGateLog(
            f: f, aLog: aLog, dtBias: dtBias.reshaped(numHeads, headDim),
            safeGate: safeGate, lowerBound: lowerBound)
        let g = exp(gLog)
        let beta = sigmoid(bProj(x).asType(.float32))

        let (out, newSsmState) = edge0GatedDeltaUpdate(
            q: q, k: k, v: v, g: g, beta: beta, state: cache?[3])
        if let cache {
            cache[3] = newSsmState
        }

        let gateR = gate.reshaped(B, T, numHeads, headDim)
        let normed = oNorm(out.asType(dtype)) * sigmoid(gateR)
        return oProj(normed.reshaped(B, T, -1))
    }
}

// MARK: - MLA (DeepSeek-style latent attention, V3 head-wise output gate)

final class Edge0BailingMLA: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qkHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let qLoraRank: Int?
    let scale: Float
    let gateKind: String?

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ args: Edge0BailingConfiguration) {
        numHeads = args.attentionHeads
        qkNopeHeadDim = args.qkNopeHeadDim
        qkRopeHeadDim = args.qkRopeHeadDim
        qkHeadDim = args.qkHeadDim
        vHeadDim = args.vHeadDim
        kvLoraRank = args.kvLoraRank
        qLoraRank = args.qLoraRank
        scale = pow(Float(qkHeadDim), -0.5)
        gateKind = args.gatedAttentionProjGranularityType

        let hidden = args.hiddenSize
        let bias = args.useQKVBias
        if let qLoraRank {
            _qAProj.wrappedValue = Linear(hidden, qLoraRank, bias: bias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: args.rmsNormEps)
            _qBProj.wrappedValue = Linear(qLoraRank, numHeads * qkHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(hidden, numHeads * qkHeadDim, bias: false)
        }
        _kvAProj.wrappedValue = Linear(hidden, kvLoraRank + qkRopeHeadDim, bias: bias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: args.rmsNormEps)
        _kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)
        if gateKind == "head_wise" {
            _gProj.wrappedValue = Linear(hidden, numHeads, bias: false)
        } else if gateKind == "element_wise" {
            _gProj.wrappedValue = Linear(hidden, numHeads * vHeadDim, bias: false)
        }
        _dense.wrappedValue = Linear(numHeads * vHeadDim, hidden, bias: bias)

        // Ling 3.0's MLA is DeepSeek-style with the same interleaved RoPE
        // convention (`traditional: true` in mlx-swift-lm's terms); see
        // DeepseekV3Attention for the reference usage this mirrors.
        rope = initializeRope(
            dims: qkRopeHeadDim, base: args.ropeTheta, traditional: true,
            scalingConfig: nil, maxPositionEmbeddings: nil)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q: MLXArray
        if qLoraRank != nil {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        } else {
            q = qProj!(x)
        }
        q = q.reshaped(B, L, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qSplit[0]
        var qPe = qSplit[1]

        var compressed = kvAProj(x)
        let compSplit = split(compressed, indices: [kvLoraRank], axis: -1)
        compressed = compSplit[0]
        var kPe = compSplit[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressed))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let kvSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvSplit[0]
        let values = kvSplit[1]

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let keys = concatenated([kNope, kPe], axis: -1)
        let queries = concatenated([qNope, qPe], axis: -1)

        var out = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if gateKind == "head_wise" {
            let gate = sigmoid(gProj!(x))
            out = out.reshaped(B, L, numHeads, vHeadDim)
            out = out * expandedDimensions(gate, axis: -1)
            out = out.reshaped(B, L, -1)
        } else if gateKind == "element_wise" {
            out = out * sigmoid(gProj!(x))
        }
        return dense(out)
    }
}

// MARK: - Dense MLP (layer 0 and shared experts)

final class Edge0BailingMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: Edge0BailingConfiguration, intermediate: Int) {
        _gateProj.wrappedValue = Linear(args.hiddenSize, intermediate, bias: false)
        _upProj.wrappedValue = Linear(args.hiddenSize, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE router (sigmoid, expert-bias, group-limited top-k)

final class Edge0BailingGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let numExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    // Optional because upstream only creates it when the config asks for it,
    // and a parameter this model declares but the checkpoint does not ship
    // fails the whole load: mlx-swift-lm applies weights with
    // `verify: [.all]`, which requires every declared parameter to be set.
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray?

    init(_ args: Edge0BailingConfiguration) {
        topK = args.numExpertsPerTok
        nGroup = args.nGroup
        topkGroup = args.topkGroup
        numExperts = args.numExperts
        routedScalingFactor = args.routedScalingFactor
        normTopkProb = args.normTopkProb
        _weight.wrappedValue = MLXArray.zeros([args.numExperts, args.hiddenSize])
        _expertBias.wrappedValue =
            args.moeRouterEnableExpertBias ? MLXArray.zeros([args.numExperts]) : nil
        super.init()
    }

    /// Selection uses `scores + expertBias` (group-drop + top-k), but the
    /// returned WEIGHTS come from the raw sigmoid `scores` — the
    /// aux-loss-free convention where the bias steers routing without
    /// biasing magnitude.
    func groupSelect(_ x: MLXArray) -> (inds: MLXArray, weights: MLXArray) {
        let bsz = x.dim(0)
        let seqLen = x.dim(1)
        let logits = matmul(x, weight.T)
        let scores = sigmoid(logits.asType(.float32))
        let selectBase = expertBias.map { scores + $0 } ?? scores

        var select = selectBase
        let kDrop = nGroup - topkGroup
        if kDrop > 0 {
            let grouped = selectBase.reshaped(bsz, seqLen, nGroup, numExperts / nGroup)
            let groupTop2Sum = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let dropIdx = argPartition(groupTop2Sum, kth: kDrop - 1, axis: -2)[
                .ellipsis, ..<kDrop, 0...]
            let masked = putAlong(
                grouped, dropIdx, values: MLXArray(-Float.infinity), axis: -2)
            select = flattened(masked, start: -2, end: -1)
        }

        let k = topK
        let idx = argPartition(-select, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var w = takeAlong(scores, idx, axis: -1)
        if normTopkProb {
            w = w / (w.sum(axis: -1, keepDims: true) + 1e-20)
        }
        w = w * routedScalingFactor
        return (idx, w.asType(logits.dtype))
    }
}

final class Edge0BailingSparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Edge0BailingGate
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Edge0BailingMLP

    init(_ args: Edge0BailingConfiguration) {
        _gate.wrappedValue = Edge0BailingGate(args)
        _experts.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedExperts.wrappedValue = Edge0BailingMLP(
            args,
            intermediate: args.moeSharedExpertIntermediateSize * args.numSharedExperts)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate.groupSelect(x)
        let routed = experts(x, inds)
        let out = (routed * weights[.ellipsis, .newAxis]).sum(axis: -2)
        return out + sharedExperts(x)
    }
}

// MARK: - Decoder layer (dispatches KDA vs MLA, dense vs MoE MLP)

final class Edge0BailingDecoderLayer: Module {
    let isMLA: Bool
    @ModuleInfo(key: "attention") var attention: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Edge0BailingConfiguration, layerIdx: Int) {
        isMLA = args.isMLALayer(layerIdx)
        if isMLA {
            _attention.wrappedValue = Edge0BailingMLA(args)
        } else {
            _attention.wrappedValue = Edge0BailingKDA(args)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if layerIdx >= args.firstKDenseReplace {
            _mlp.wrappedValue = Edge0BailingSparseMoE(args)
        } else {
            _mlp.wrappedValue = Edge0BailingMLP(args, intermediate: args.intermediateSize)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let attnOut: MLXArray
        if isMLA {
            attnOut = (attention as! Edge0BailingMLA)(inputLayerNorm(x), mask: mask, cache: cache)
        } else {
            attnOut = (attention as! Edge0BailingKDA)(
                inputLayerNorm(x), cache: cache as? ArraysCache)
        }
        let h = x + attnOut
        let normed = postAttentionLayerNorm(h)
        let mlpOut: MLXArray
        if let moe = mlp as? Edge0BailingSparseMoE {
            mlpOut = moe(normed)
        } else {
            mlpOut = (mlp as! Edge0BailingMLP)(normed)
        }
        return h + mlpOut
    }
}

// MARK: - Model

final class Edge0BailingModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    let layers: [Edge0BailingDecoderLayer]
    let norm: RMSNorm
    let firstMLAIdx: Int

    init(_ args: Edge0BailingConfiguration) {
        precondition(args.vocabularySize > 0)
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        layers = (0 ..< args.hiddenLayers).map { Edge0BailingDecoderLayer(args, layerIdx: $0) }
        norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        firstMLAIdx = (0 ..< args.hiddenLayers).first { args.isMLALayer($0) } ?? 0
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = wordEmbeddings(inputs)
        let mlaMask = createAttentionMask(h: h, cache: cache?[firstMLAIdx])
        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = layer.isMLA ? mlaMask : .none
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class Edge0BailingModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let model: Edge0BailingModelInner
    let configuration: Edge0BailingConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Edge0BailingConfiguration) {
        configuration = args
        vocabularySize = args.vocabularySize
        kvHeads = (0 ..< args.hiddenLayers).map { _ in args.attentionHeads }
        model = Edge0BailingModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.wordEmbeddings.asLinear(out)
    }

    /// KDA layers keep four rolling state arrays (q/k/v short-conv states plus
    /// the delta-rule recurrent state); MLA layers use a normal KV cache.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { layer in
            layer.isMLA ? KVCacheSimple() : ArraysCache(size: 4)
        }
    }

    public func makeCache() -> [KVCache] {
        newCache(parameters: nil)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        // The stock loader globs every safetensors file in the model
        // directory, which for edge0 includes the LoRA and prerouter adapters.
        // They are applied separately (or not at all); left here they reach
        // `update(parameters:verify:[.all])` as keys the model does not have.
        for key in Array(weights.keys) where Edge0Adapters.isAdapterTensor(key) {
            weights.removeValue(forKey: key)
        }

        for key in Array(weights.keys) where key.contains(".mtp_") || key.contains("mtp.") {
            weights.removeValue(forKey: key)
        }

        if configuration.tieWordEmbeddings {
            for key in Array(weights.keys)
            where key.split(separator: ".").contains("lm_head") {
                weights.removeValue(forKey: key)
            }
        }

        // Stack per-expert weights into SwitchGLU's [numExperts, out, in] layout.
        let numExperts = configuration.numExperts
        for li in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(li)"
            let probeWeight = "\(prefix).mlp.experts.0.gate_proj.weight"
            let probeScales = "\(prefix).mlp.experts.0.gate_proj.scales"
            if weights[probeWeight] == nil && weights[probeScales] == nil {
                continue
            }
            for m in ["gate_proj", "up_proj", "down_proj"] {
                for part in ["weight", "scales", "biases"] {
                    let key0 = "\(prefix).mlp.experts.0.\(m).\(part)"
                    guard weights[key0] != nil else { continue }
                    var pieces: [MLXArray] = []
                    pieces.reserveCapacity(numExperts)
                    for e in 0 ..< numExperts {
                        guard
                            let v = weights.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(m).\(part)")
                        else { continue }
                        pieces.append(v)
                    }
                    if pieces.count == numExperts {
                        weights["\(prefix).mlp.experts.\(m).\(part)"] = MLX.stacked(pieces)
                    }
                }
            }
        }

        // Conv weights: checkpoint stores torch depthwise layout [C, 1, ksize]
        // (or flat [C, ksize]); Edge0ShortConv1d's nested Conv1d expects
        // [C, ksize, 1].
        for key in Array(weights.keys) where key.hasSuffix("_conv1d.weight") {
            guard var w = weights.removeValue(forKey: key) else { continue }
            if w.ndim == 2 {
                w = expandedDimensions(w, axis: -1)
            } else if w.ndim == 3 && w.dim(-1) != 1 {
                w = w.movedAxis(source: 2, destination: 1)
            }
            let newKey = String(key.dropLast(".weight".count)) + ".conv.weight"
            weights[newKey] = w
        }

        return weights
    }
}

extension Edge0BailingModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
