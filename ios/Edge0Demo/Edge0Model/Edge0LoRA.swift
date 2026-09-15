// Loads edge0's "parallel (unmerged) LoRA adapter" safetensors — a port of
// edge0/src/edge0/adapters/lora.py's `install_lora` (Edge0-AI/edge0,
// Apache-2.0) — onto a loaded model.
//
// Rather than doing the module surgery by hand (which would also have to
// special-case quantized bases), this drives mlx-swift-lm's own
// `LoRAContainer`: it wraps each target `Linear`/`QuantizedLinear` in
// `LoRALinear`/`QLoRALinear` and then loads adapter parameters by path. That
// keeps the 4-bit base weights quantized and works for any `LoRAModel` — both
// the hand-ported 8B backbone and mlx-swift-lm's native Qwen3.5-MoE used by
// the 35B tier.
//
// Two conversions bridge edge0's file format to that machinery:
//   * `lora_A [r, in]` / `lora_B [out, r]`  ->  `lora_a [in, r]` / `lora_b [r, out]`
//   * `scale = alpha / r`, matching edge0's own `scale = alpha / r`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct Edge0LoRAReport: Sendable {
    var appliedTargets: [String] = []
    var unmatchedTargets: [String] = []
    var scale: Float = 0
    /// Rank actually found in the adapter file, when it disagrees with the
    /// rank the scale was computed from. `scale = alpha / r`, so a disagreement
    /// means every delta is off by a constant factor — which looks like a model
    /// that is simply a bit worse, not like a bug.
    var rankMismatch: Int?

    var summary: String {
        var text = "applied=\(appliedTargets.count) unmatched=\(unmatchedTargets.count)"
            + " scale=\(scale)"
        if let rankMismatch {
            text += " WARNING: adapter rank is \(rankMismatch), scale assumes another"
        }
        return text
    }
}

enum Edge0LoRAError: LocalizedError {
    case notALoRAModel
    case noAdapterTensors

    var errorDescription: String? {
        switch self {
        case .notALoRAModel: "Model does not support LoRA adapters."
        case .noAdapterTensors: "Adapter file contains no lora_A/lora_B tensors."
        }
    }
}

enum Edge0LoRA {

    /// Applies `fileURL`'s adapters to `model` in place.
    @discardableResult
    static func apply(
        to model: any LanguageModel, fileURL: URL, rank: Int, alpha: Float
    ) throws -> Edge0LoRAReport {
        guard let loraModel = model as? LoRAModel else { throw Edge0LoRAError.notALoRAModel }

        // 1. Group the adapter file into `<target>: (A, B)`.
        let raw = try loadArrays(url: fileURL)
        var adapters: [String: (a: MLXArray?, b: MLXArray?)] = [:]
        for (key, value) in raw {
            if key.hasSuffix(".lora_A") {
                adapters[String(key.dropLast(7)), default: (nil, nil)].a = value
            } else if key.hasSuffix(".lora_B") {
                adapters[String(key.dropLast(7)), default: (nil, nil)].b = value
            }
        }
        guard !adapters.isEmpty else { throw Edge0LoRAError.noAdapterTensors }

        // 2. Index the model's real linear modules by (layer index, path within
        //    the layer). The adapter file and the Swift module tree can disagree
        //    on the prefix (edge0's 35B keys start with `language_model.`), so
        //    the layer index plus the in-layer path is what actually matches.
        var linearPaths: [LayerLocalKey: String] = [:]
        for (path, module) in model.namedModules() {
            guard module is Linear, let key = LayerLocalKey(path: path) else { continue }
            linearPaths[key] = path
        }

        // 3. Resolve each adapter target against that index.
        var report = Edge0LoRAReport(scale: alpha / Float(rank))
        var parameters: [String: MLXArray] = [:]
        var relativeKeys: Set<String> = []

        for (target, pair) in adapters.sorted(by: { $0.key < $1.key }) {
            guard let a = pair.a, let b = pair.b, let key = LayerLocalKey(path: target),
                let modulePath = linearPaths[key]
            else {
                report.unmatchedTargets.append(target)
                continue
            }
            // edge0 computes the scale from its configured rank rather than
            // from the file, so this follows suit — but records a disagreement
            // instead of letting it pass unnoticed.
            if a.ndim == 2, a.dim(0) != rank, report.rankMismatch == nil {
                report.rankMismatch = a.dim(0)
            }
            parameters["\(modulePath).lora_a"] = a.T
            parameters["\(modulePath).lora_b"] = b.T
            relativeKeys.insert(key.relativePath)
            report.appliedTargets.append(modulePath)
        }

        guard !parameters.isEmpty else { return report }

        // 4. Hand it to mlx-swift-lm: wrap the targets, then load the weights.
        let configuration = LoRAConfiguration(
            numLayers: loraModel.loraLayers.count,
            fineTuneType: .lora,
            loraParameters: .init(
                rank: rank, scale: report.scale, keys: Array(relativeKeys))
        )
        let container = LoRAContainer(
            configuration: configuration,
            parameters: ModuleParameters.unflattened(parameters)
        )
        try container.load(into: model)

        print("[edge0-lora] \(report.summary)")
        return report
    }
}

/// A module path split at its `layers.<n>.` boundary — e.g.
/// `language_model.model.layers.7.self_attn.q_proj` becomes layer 7 +
/// `self_attn.q_proj`.
private struct LayerLocalKey: Hashable {
    let layerIndex: Int
    let relativePath: String

    init?(path: String) {
        let parts = path.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count >= 3 else { return nil }
        // Scan from the end so a model whose own prefix also contains "layers"
        // still splits at the decoder-layer boundary closest to the leaf.
        var cursor = parts.count - 3
        var match: (index: Int, layer: Int)?
        while cursor >= 0 {
            if parts[cursor] == "layers", let layer = Int(parts[cursor + 1]) {
                match = (cursor, layer)
                break
            }
            cursor -= 1
        }
        guard let match else { return nil }
        layerIndex = match.layer
        relativePath = parts[(match.index + 2)...].joined(separator: ".")
    }
}
