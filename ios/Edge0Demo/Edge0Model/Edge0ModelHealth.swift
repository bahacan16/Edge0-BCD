// A one-token forward pass run right after loading.
//
// The 8B backbone is a hand-written port, and the 35B path swaps its expert
// layer for a streaming one — both are places where a shape or dtype mistake
// shows up as NaN logits rather than as an error. Without this check the first
// symptom the user sees is gibberish; with it, loading fails with something
// they can report.

import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct Edge0HealthReport: Sendable {
    var passed: Bool
    var vocabularySize: Int
    var detail: String

    static let skipped = Edge0HealthReport(
        passed: true, vocabularySize: 0, detail: "atlandı")
}

enum Edge0ModelHealth {

    /// Runs one token through the model and checks the logits are finite.
    static func check(
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer
    ) -> Edge0HealthReport {
        let probe = tokenizer.encode(text: "Merhaba").first ?? 1
        let cache = model.newCache(parameters: nil)
        let logits = model(MLXArray([Int32(probe)]).reshaped(1, 1), cache: cache)
        eval(logits)

        let vocabulary = logits.dim(-1)
        guard vocabulary > 0 else {
            return Edge0HealthReport(
                passed: false, vocabularySize: 0, detail: "model boş logit üretti")
        }

        let finite = MLX.all(MLX.isFinite(logits.asType(.float32))).item(Bool.self)
        guard finite else {
            return Edge0HealthReport(
                passed: false, vocabularySize: vocabulary,
                detail: "logitler NaN/Inf — ağırlık eşlemesi veya sayısal bir hata var")
        }

        return Edge0HealthReport(
            passed: true, vocabularySize: vocabulary, detail: "tamam")
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
