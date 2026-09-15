// VENDORED from ml-explore/mlx-swift-lm Libraries/MLXLLM/Models/Qwen3Next.swift (MIT License).
//
// Why a copy instead of the package's own types: the 35B tier cannot hold its
// experts in RAM, so its `switch_mlp` has to be replaced with a streaming
// implementation. mlx-swift-lm 3.31.4 declares `SwitchGLU` (and the model
// classes around it) as `public`, not `open`, so they cannot be subclassed
// from another module. Vendoring the file makes those types local, and the
// only behavioral change is that the MoE block's expert layer is swappable.
//
// Types are prefixed `E0` so they never collide with the package's own.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

func sigmoidMultiply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
    x * sigmoid(gate)
}

private func preciseSwiGLU(_ hiddenStates: MLXArray, gate: MLXArray, x: MLXArray) -> MLXArray {
    (silu(gate.asType(.float32)) * x.asType(.float32)).asType(hiddenStates.dtype)
}

// MARK: - Model Components


final class E0Qwen3NextRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        if let gate {
            x = preciseSwiGLU(hiddenStates, gate: gate, x: x)
        }
        return x
    }
}


final class E0Qwen3NextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

