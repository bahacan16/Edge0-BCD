// A short greedy generation run right after loading.
//
// The 8B backbone is a hand-written port, and the 35B path swaps its expert
// layer for a streaming one — both are places where a shape or dtype mistake
// produces a model that loads cleanly and then talks nonsense. Without a check
// the first symptom the user sees is gibberish half an hour into a chat; with
// one, loading reports something they can act on.
//
// Two levels, deliberately:
//   * NaN/Inf logits are a hard failure — there is no reading of that which is
//     working software.
//   * A degenerate sample is only reported, not failed, because greedy decoding
//     legitimately gets stuck sometimes and refusing to load a working model
//     would be the worse bug. The one exception is a run that produces the same
//     token twelve times, which no working chat model does on a real prompt.

import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct Edge0HealthReport: Sendable {
    var passed: Bool
    var vocabularySize: Int
    var detail: String
    /// What the model actually said, for the Settings readout.
    var sample: String = ""

    static let skipped = Edge0HealthReport(
        passed: true, vocabularySize: 0, detail: "atlandı")
}

enum Edge0ModelHealth {
    private static let probe = "Merhaba, kısaca kendini tanıt."
    private static let steps = 12

    static func check(
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer
    ) -> Edge0HealthReport {
        var prompt = tokenizer.encode(text: probe)
        if prompt.isEmpty { prompt = [1] }

        let cache = model.newCache(parameters: nil)
        var logits = model(
            MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count), cache: cache)
        eval(logits)

        let vocabulary = logits.dim(-1)
        guard vocabulary > 0 else {
            return Edge0HealthReport(
                passed: false, vocabularySize: 0, detail: "model boş logit üretti")
        }

        guard isFinite(logits) else {
            return Edge0HealthReport(
                passed: false, vocabularySize: vocabulary,
                detail: "logitler NaN/Inf — ağırlık eşlemesi veya sayısal bir hata var")
        }

        // Greedy, so this is deterministic and says something about the weights
        // rather than about the sampler.
        var generated: [Int] = []
        for _ in 0 ..< steps {
            let next = MLX.argMax(logits[.ellipsis, -1, 0...], axis: -1)
            eval(next)
            generated.append(next.item(Int.self))
            logits = model(next.reshaped(1, 1), cache: cache)
            eval(logits)
            guard isFinite(logits) else {
                return Edge0HealthReport(
                    passed: false, vocabularySize: vocabulary,
                    detail: "üretim sırasında logitler NaN/Inf oldu")
            }
        }

        let sample = tokenizer.decode(tokens: generated)
        let distinct = Set(generated).count

        if distinct <= 1 {
            return Edge0HealthReport(
                passed: false, vocabularySize: vocabulary,
                detail: "model tek bir tokenı tekrarlıyor — ağırlıklar hatalı eşlenmiş olabilir",
                sample: sample)
        }

        return Edge0HealthReport(
            passed: true, vocabularySize: vocabulary,
            detail: distinct < 4 ? "geçti (çıktı çok tekrarlı)" : "tamam",
            sample: sample)
    }

    private static func isFinite(_ logits: MLXArray) -> Bool {
        MLX.all(MLX.isFinite(logits.asType(.float32))).item(Bool.self)
    }
}

enum Edge0HealthError: LocalizedError {
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .failed(let detail): "Model sağlık kontrolünden geçemedi: \(detail)"
        }
    }
}
