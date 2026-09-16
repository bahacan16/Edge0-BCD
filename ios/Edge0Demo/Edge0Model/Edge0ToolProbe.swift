// Does this checkpoint do tool calls?
//
// Two separate questions, and they have to be answered in order, because the
// second one is meaningless if the first is no:
//
//   1. Does the chat template render tools at all? A template that ignores the
//      `tools` variable produces the same prompt with and without it, and no
//      amount of prompting gets the model to call something it was never told
//      about.
//   2. Given a prompt that does declare a tool, does the model emit a call —
//      and in what shape? Qwen writes `<tool_call>{json}</tool_call>`, Ling
//      writes something of its own, and a parser written for the wrong one
//      silently sees plain prose.
//
// Both are answered here against the real checkpoint on the device, because
// the alternative is inferring them from a model card. edge0's own engine
// renders its template with `tools=None` and a per-message `tool_calls` field,
// which says the template has the machinery — but "has the machinery" and
// "this build can drive it" are not the same claim, and only one of them can
// be tested.

import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct Edge0ToolProbeReport: Sendable {
    /// The template produced a different prompt when handed a tool.
    var templateAcceptsTools = false
    /// The tool's name reached the prompt, rather than the prompt merely
    /// growing for some other reason.
    var toolNameInPrompt = false
    /// Tokens the declaration cost, which is what every tool-enabled turn pays
    /// before the question is even read.
    var promptTokens = 0
    var baselineTokens = 0
    /// What the model said when asked something the tool answers.
    var sample = ""
    /// The call shape found in `sample`, when one was.
    var callMarker: String?
    var error: String?

    var verdict: String {
        if let error { return "çalıştırılamadı: \(error)" }
        if !templateAcceptsTools { return "şablon araçları yok sayıyor" }
        if let callMarker { return "araç çağrısı üretti (\(callMarker))" }
        return "şablon araçları tanıyor, model bu istemde çağrı üretmedi"
    }
}

enum Edge0ToolProbe {
    private static let question = "İstanbul'da hava durumu nedir?"
    /// Generous, and stopped early the moment a call appears.
    ///
    /// Forty-eight was not: both tiers spent every one of those tokens
    /// *reasoning about* the call and got cut off before making it. The 35B's
    /// was unambiguous — "I have a tool `get_weather` ... I should call this
    /// tool with İstanbul as the city" — which is a model doing the right thing
    /// and a probe too impatient to watch it finish.
    private static let steps = 320

    /// A minimal OpenAI-shaped function, which is the shape every template that
    /// supports tools expects.
    private static var weatherTool: [String: any Sendable] {
        let parameters: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "city": ["type": "string", "description": "Şehir adı"] as [String: any Sendable]
            ] as [String: any Sendable],
            "required": ["city"],
        ]
        return [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Bir şehrin güncel hava durumunu döndürür.",
                "parameters": parameters,
            ] as [String: any Sendable],
        ]
    }

    /// Markers the known families wrap a call in. Found in the output, these
    /// say which parser this checkpoint would need.
    private static let markers = [
        "<tool_call>", "<|tool_call|>", "<function_call>", "<|action_start|>",
        "<tool▁call▁begin|>", "<tools>", "```json",
    ]

    /// A bare call with no wrapper around it — some templates ask for exactly
    /// this, and a marker list alone would score it as prose.
    private static func looksLikeBareCall(_ text: String) -> Bool {
        text.contains("\"name\"") && text.contains("get_weather")
            && (text.contains("\"arguments\"") || text.contains("\"parameters\""))
    }

    static func callShape(in text: String) -> String? {
        if let marker = markers.first(where: { text.contains($0) }) { return marker }
        return looksLikeBareCall(text) ? "sarmalayıcısız JSON" : nil
    }

    static func run(model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer)
        -> Edge0ToolProbeReport
    {
        var report = Edge0ToolProbeReport()
        let messages: [[String: any Sendable]] = [["role": "user", "content": question]]

        do {
            // Thinking off. Both tiers reason before answering, and a reasoning
            // preamble is not the thing being measured — it is the thing that
            // hid the answer the first time this ran.
            let context: [String: any Sendable] = ["enable_thinking": false]
            let baseline = try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: context)
            let tooled = try tokenizer.applyChatTemplate(
                messages: messages, tools: [weatherTool], additionalContext: context)

            report.baselineTokens = baseline.count
            report.promptTokens = tooled.count
            report.templateAcceptsTools = tooled.count != baseline.count
            report.toolNameInPrompt =
                tokenizer.decode(tokenIds: tooled).contains("get_weather")

            // No point generating against a prompt that never mentioned the
            // tool: whatever came back would say nothing about tool calling.
            guard report.templateAcceptsTools, report.toolNameInPrompt else { return report }

            report.sample = greedy(model: model, tokenizer: tokenizer, prompt: tooled)
            report.callMarker = callShape(in: report.sample)
        } catch {
            report.error = error.localizedDescription
        }
        return report
    }

    /// Greedy so the answer is about the checkpoint rather than about the
    /// sampler — a tool call that only appears at temperature 0.7 one time in
    /// three is not a capability anything can be built on.
    private static func greedy(
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer, prompt: [Int]
    ) -> String {
        let cache = model.newCache(parameters: nil)
        var logits = model(
            MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count), cache: cache)
        eval(logits)

        var generated: [Int] = []
        for step in 0 ..< steps {
            let next = MLX.argMax(logits[.ellipsis, -1, 0...], axis: -1)
            eval(next)
            let token = next.item(Int.self)
            if let eos = tokenizer.eosToken, tokenizer.convertTokenToId(eos) == token { break }
            generated.append(token)

            // Checked as it goes, so a checkpoint that calls immediately costs
            // a second rather than the full run. Every sixteen tokens because
            // decoding is cheap next to a forward pass but not free, and a call
            // marker is several tokens wide anyway.
            if step % 16 == 15,
                callShape(in: tokenizer.decode(tokenIds: generated)) != nil
            {
                break
            }

            logits = model(next.reshaped(1, 1), cache: cache)
            eval(logits)
        }
        return tokenizer.decode(tokenIds: generated)
    }
}
