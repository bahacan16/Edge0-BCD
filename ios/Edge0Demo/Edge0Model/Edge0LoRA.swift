// Applies edge0's "parallel (unmerged) LoRA adapter" safetensors — a port
// of edge0/src/edge0/adapters/lora.py's `install_lora` (Edge0-AI/edge0,
// Apache-2.0) — onto an already-loaded Edge0BailingModel, reusing
// mlx-swift-lm's built-in `LoRALinear` (MLXLMCommon/Adapters/LoRA) for the
// actual `y = base(x) + scale * (x @ A^T) @ B^T` math instead of
// reimplementing it: `LoRALinear`'s `loraA`/`loraB` are the transpose of
// edge0's raw `lora_A [r, in]` / `lora_B [out, r]` tensors, so loading is
// just "transpose then assign", with `scale = alpha / r` matching exactly.
//
// Swift has no Python-style `getattr`/`setattr` by dotted string path, so
// unlike the Python original (fully generic over whatever keys the
// safetensors file happens to contain) this walks an explicit, enumerated
// set of Linear-typed fields per attention flavor (KDA vs MLA) and per MLP
// flavor (dense vs the MoE block's shared experts) — every field
// `install_lora`'s own `_resolve` can reach on this architecture.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct Edge0LoRAApplyResult {
    var applied: [String] = []
    var notFound: [String] = []
}

private func loraDelta(
    _ weights: [String: MLXArray], prefix: String, alpha: Float, r: Int
) -> (a: MLXArray, b: MLXArray, scale: Float)? {
    guard let a = weights["\(prefix).lora_A"], let b = weights["\(prefix).lora_B"] else {
        return nil
    }
    // edge0: A [r, in], B [out, r]; LoRALinear wants loraA [in, r], loraB [r, out].
    return (a.T, b.T, alpha / Float(r))
}

/// Wraps `current` (if it's a plain `Linear`, not already LoRA-wrapped) with
/// the adapter found at `prefix`, or returns `current` unchanged.
private func applyIfPresent(
    _ current: Linear, prefix: String, weights: [String: MLXArray], alpha: Float, r: Int,
    result: inout Edge0LoRAApplyResult
) -> Linear {
    guard let (a, b, scale) = loraDelta(weights, prefix: prefix, alpha: alpha, r: r) else {
        result.notFound.append(prefix)
        return current
    }
    let (outputDimensions, inputDimensions) = current.shape
    let wrapped = LoRALinear(
        inputDimensions, outputDimensions, rank: r, scale: scale, dropout: 0.0, linear: current)
    wrapped._loraA.wrappedValue = a.asType(current.weight.dtype)
    wrapped._loraB.wrappedValue = b.asType(current.weight.dtype)
    result.applied.append(prefix)
    return wrapped
}

/// Loads `lora_edge0_8b.safetensors`-style adapters (from `fileURL`) and
/// wraps every matching Linear projection in `model` in place.
func applyEdge0LoRA(
    model: Edge0BailingModel, fileURL: URL, r: Int = 16, alpha: Float = 32.0
) throws -> Edge0LoRAApplyResult {
    let weights = try loadArrays(url: fileURL)
    var result = Edge0LoRAApplyResult()

    for (li, layer) in model.model.layers.enumerated() {
        let attnPrefix = "model.layers.\(li).attention"
        let mlpPrefix = "model.layers.\(li).mlp"

        if layer.isMLA, let mla = layer.attention as? Edge0BailingMLA {
            if let q = mla.qProj {
                mla.qProj = applyIfPresent(
                    q, prefix: "\(attnPrefix).q_proj", weights: weights, alpha: alpha, r: r,
                    result: &result)
            }
            if let qa = mla.qAProj {
                mla.qAProj = applyIfPresent(
                    qa, prefix: "\(attnPrefix).q_a_proj", weights: weights, alpha: alpha, r: r,
                    result: &result)
            }
            if let qb = mla.qBProj {
                mla.qBProj = applyIfPresent(
                    qb, prefix: "\(attnPrefix).q_b_proj", weights: weights, alpha: alpha, r: r,
                    result: &result)
            }
            mla.kvAProj = applyIfPresent(
                mla.kvAProj, prefix: "\(attnPrefix).kv_a_proj_with_mqa", weights: weights,
                alpha: alpha, r: r, result: &result)
            mla.kvBProj = applyIfPresent(
                mla.kvBProj, prefix: "\(attnPrefix).kv_b_proj", weights: weights, alpha: alpha,
                r: r, result: &result)
            mla.dense = applyIfPresent(
                mla.dense, prefix: "\(attnPrefix).dense", weights: weights, alpha: alpha, r: r,
                result: &result)
        } else if let kda = layer.attention as? Edge0BailingKDA {
            kda.qProj = applyIfPresent(
                kda.qProj, prefix: "\(attnPrefix).q_proj", weights: weights, alpha: alpha, r: r,
                result: &result)
            kda.kProj = applyIfPresent(
                kda.kProj, prefix: "\(attnPrefix).k_proj", weights: weights, alpha: alpha, r: r,
                result: &result)
            kda.vProj = applyIfPresent(
                kda.vProj, prefix: "\(attnPrefix).v_proj", weights: weights, alpha: alpha, r: r,
                result: &result)
            kda.oProj = applyIfPresent(
                kda.oProj, prefix: "\(attnPrefix).o_proj", weights: weights, alpha: alpha, r: r,
                result: &result)
        }

        if let dense = layer.mlp as? Edge0BailingMLP {
            dense.gateProj = applyIfPresent(
                dense.gateProj, prefix: "\(mlpPrefix).gate_proj", weights: weights, alpha: alpha,
                r: r, result: &result)
            dense.upProj = applyIfPresent(
                dense.upProj, prefix: "\(mlpPrefix).up_proj", weights: weights, alpha: alpha,
                r: r, result: &result)
            dense.downProj = applyIfPresent(
                dense.downProj, prefix: "\(mlpPrefix).down_proj", weights: weights, alpha: alpha,
                r: r, result: &result)
        } else if let moe = layer.mlp as? Edge0BailingSparseMoE {
            let sharedPrefix = "\(mlpPrefix).shared_experts"
            moe.sharedExperts.gateProj = applyIfPresent(
                moe.sharedExperts.gateProj, prefix: "\(sharedPrefix).gate_proj",
                weights: weights, alpha: alpha, r: r, result: &result)
            moe.sharedExperts.upProj = applyIfPresent(
                moe.sharedExperts.upProj, prefix: "\(sharedPrefix).up_proj", weights: weights,
                alpha: alpha, r: r, result: &result)
            moe.sharedExperts.downProj = applyIfPresent(
                moe.sharedExperts.downProj, prefix: "\(sharedPrefix).down_proj",
                weights: weights, alpha: alpha, r: r, result: &result)
        }
    }

    print(
        "[edge0-lora] applied=\(result.applied.count) not_found=\(result.notFound.count)")
    return result
}
