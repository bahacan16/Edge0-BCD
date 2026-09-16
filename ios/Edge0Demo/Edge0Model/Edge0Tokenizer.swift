// Tokenizer loading that honours a `chat_template.jinja` sidecar.
//
// Newer Hugging Face exports write the chat template to its own file rather
// than embedding it in tokenizer_config.json, and edge0's repositories are
// exported that way — the 35B repo ships a 7.76 kB chat_template.jinja next to
// a 1.14 kB tokenizer_config.json.
//
// swift-transformers only ever looks for `chat_template` inside the tokenizer
// config, so on those checkpoints `applyChatTemplate` throws. That failure is
// caught upstream of here and falls back to joining the messages as plain
// text, which loads and generates and is subtly, unfixably wrong: an
// instruction-tuned model handed an unformatted transcript has no turn markers
// to answer into. The symptom is a model that works but answers badly — the
// hardest kind of wrong to attribute.

import Foundation
import MLXLMCommon
import Tokenizers

/// Mirrors what `#huggingFaceTokenizerLoader()` builds, plus the sidecar.
struct Edge0TokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        let sidecar = try? String(
            contentsOf: directory.appending(component: "chat_template.jinja"), encoding: .utf8)
        return Edge0Tokenizer(upstream: upstream, sidecarTemplate: sidecar)
    }
}

struct Edge0Tokenizer: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer
    /// The template from `chat_template.jinja`, when the checkpoint ships one.
    let sidecarTemplate: String?

    /// Whether a chat template was found at all, by either route.
    var hasChatTemplate: Bool {
        if sidecarTemplate != nil { return true }
        do {
            _ = try upstream.applyChatTemplate(messages: [["role": "user", "content": "x"]])
            return true
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            return false
        } catch {
            // Some other template problem: present, just unhappy with the
            // probe. Not this check's business.
            return true
        }
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    // swift-transformers spells this `decode(tokens:)`.
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            guard let sidecarTemplate else {
                throw MLXLMCommon.TokenizerError.missingChatTemplate
            }
            return try upstream.applyChatTemplate(
                messages: messages,
                chatTemplate: .literal(sidecarTemplate),
                // This overload defaults `addGenerationPrompt` to false, unlike
                // the one above it. Without it the model receives the
                // conversation and is never prompted to answer.
                addGenerationPrompt: true,
                tools: tools,
                additionalContext: additionalContext)
        }
    }
}
