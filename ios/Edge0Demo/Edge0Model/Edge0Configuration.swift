// Port of edge0/src/edge0/backends/mlx/_impl/bailing_hybrid.py ModelArgs
// (Edge0-AI/edge0, Apache-2.0) — the Ling 3.0 "bailing_hybrid" backbone
// used by the edge0-8b tier (KDA linear-attention layers + MLA layers +
// sigmoid group-limited-topk MoE).

import Foundation
import MLXLMCommon

public struct Edge0BailingConfiguration: Codable, Sendable {
    var modelType: String = "bailing_hybrid"
    var hiddenSize: Int = 1536
    var hiddenLayers: Int = 24
    var intermediateSize: Int = 4608
    var attentionHeads: Int = 16
    var kvHeads: Int = 16
    var headDim: Int = 128
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 157184
    var tieWordEmbeddings: Bool = false

    // Layer schedule
    var layerGroupSize: Int = 4
    var firstKDenseReplace: Int = 1

    // KDA (linear attention)
    var shortConvKernelSize: Int = 4
    var noKdaLora: Bool = true
    var kdaSafeGate: Bool = true
    var kdaLowerBound: Float = -5.0

    // MLA
    var qLoraRank: Int? = 256
    var kvLoraRank: Int = 512
    var qkNopeHeadDim: Int = 128
    var qkRopeHeadDim: Int = 64
    var vHeadDim: Int = 128
    var ropeTheta: Float = 6_000_000.0
    var useQKVBias: Bool = false
    var gatedAttentionProjGranularityType: String? = "head_wise"

    // MoE
    var numExperts: Int = 128
    var numExpertsPerTok: Int = 8
    var numSharedExperts: Int = 1
    var moeIntermediateSize: Int = 512
    var moeSharedExpertIntermediateSize: Int = 512
    var nGroup: Int = 8
    var topkGroup: Int = 4
    var normTopkProb: Bool = true
    var routedScalingFactor: Float = 2.5
    var moeRouterEnableExpertBias: Bool = true

    var qkHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    /// Softmax (MLA) layers sit last in each group; any remainder layers
    /// past the final full group are MLA too.
    func isMLALayer(_ idx: Int) -> Bool {
        let g = layerGroupSize
        let full = (hiddenLayers / g) * g
        return (idx + 1) % g == 0 || idx >= full
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerGroupSize = "layer_group_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case shortConvKernelSize = "short_conv_kernel_size"
        case noKdaLora = "no_kda_lora"
        case kdaSafeGate = "kda_safe_gate"
        case kdaLowerBound = "kda_lower_bound"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeTheta = "rope_theta"
        case useQKVBias = "use_qkv_bias"
        case gatedAttentionProjGranularityType = "gated_attention_proj_granularity_type"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func get<T: Decodable>(_ key: CodingKeys, _ def: T) -> T {
            (try? c.decodeIfPresent(T.self, forKey: key)) ?? def
        }
        modelType = get(.modelType, "bailing_hybrid")
        hiddenSize = get(.hiddenSize, 1536)
        hiddenLayers = get(.hiddenLayers, 24)
        intermediateSize = get(.intermediateSize, 4608)
        attentionHeads = get(.attentionHeads, 16)
        kvHeads = get(.kvHeads, 16)
        headDim = get(.headDim, 128)
        rmsNormEps = get(.rmsNormEps, 1e-6)
        vocabularySize = get(.vocabularySize, 157184)
        tieWordEmbeddings = get(.tieWordEmbeddings, false)
        layerGroupSize = get(.layerGroupSize, 4)
        firstKDenseReplace = get(.firstKDenseReplace, 1)
        shortConvKernelSize = get(.shortConvKernelSize, 4)
        noKdaLora = get(.noKdaLora, true)
        kdaSafeGate = get(.kdaSafeGate, true)
        kdaLowerBound = get(.kdaLowerBound, -5.0)
        qLoraRank = get(.qLoraRank, 256)
        kvLoraRank = get(.kvLoraRank, 512)
        qkNopeHeadDim = get(.qkNopeHeadDim, 128)
        qkRopeHeadDim = get(.qkRopeHeadDim, 64)
        vHeadDim = get(.vHeadDim, 128)
        ropeTheta = get(.ropeTheta, 6_000_000.0)
        useQKVBias = get(.useQKVBias, false)
        gatedAttentionProjGranularityType = get(.gatedAttentionProjGranularityType, "head_wise")
        numExperts = get(.numExperts, 128)
        numExpertsPerTok = get(.numExpertsPerTok, 8)
        numSharedExperts = get(.numSharedExperts, 1)
        moeIntermediateSize = get(.moeIntermediateSize, 512)
        moeSharedExpertIntermediateSize = get(.moeSharedExpertIntermediateSize, 512)
        nGroup = get(.nGroup, 8)
        topkGroup = get(.topkGroup, 4)
        normTopkProb = get(.normTopkProb, true)
        routedScalingFactor = get(.routedScalingFactor, 2.5)
        moeRouterEnableExpertBias = get(.moeRouterEnableExpertBias, true)
    }
}
