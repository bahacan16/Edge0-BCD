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
    private static let question = "İstanbul'da hava nasıl? Gerekiyorsa aracı kullan."
    private static let steps = 48

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
        "```json", "<tools>",
    ]

    static func run(model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer)
        -> Edge0ToolProbeReport
    {
        var report = Edge0ToolProbeReport()
        let messages: [[String: any Sendable]] = [["role": "user", "content": question]]

        do {
            let baseline = try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: nil)
            let tooled = try tokenizer.applyChatTemplate(
                messages: messages, tools: [weatherTool], additionalContext: nil)

            report.baselineTokens = baseline.count
            report.promptTokens = tooled.count
            report.templateAcceptsTools = tooled.count != baseline.count
            report.toolNameInPrompt =
                tokenizer.decode(tokenIds: tooled).contains("get_weather")

            // No point generating against a prompt that never mentioned the
            // tool: whatever came back would say nothing about tool calling.
            guard report.templateAcceptsTools, report.toolNameInPrompt else { return report }

            report.sample = greedy(model: model, tokenizer: tokenizer, prompt: tooled)
            report.callMarker = markers.first { report.sample.contains($0) }
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
        for _ in 0 ..< steps {
            let next = MLX.argMax(logits[.ellipsis, -1, 0...], axis: -1)
            eval(next)
            let token = next.item(Int.self)
            if let eos = tokenizer.eosToken, tokenizer.convertTokenToId(eos) == token { break }
            generated.append(token)
            logits = model(next.reshaped(1, 1), cache: cache)
            eval(logits)
        }
        return tokenizer.decode(tokenIds: generated)
    }
}
