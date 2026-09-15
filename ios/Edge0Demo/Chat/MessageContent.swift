// Turns a raw model answer into renderable blocks.
//
// Both tiers are reasoning models: with thinking mode on they emit a
// `<think>…</think>` preamble before the answer, and they write the answer
// itself in Markdown. Dropping that straight into a `Text` shows the user the
// literal tags and unrendered code fences, so the bubble parses it first.
//
// The parser has to cope with a half-arrived stream, where the closing tag or
// the closing fence simply has not been generated yet — an unterminated block
// runs to the end of what has arrived rather than being dropped.

import Foundation

struct MessageBlock: Identifiable, Equatable {
    enum Kind: Equatable {
        case heading(level: Int)
        case paragraph
        /// A list item. `marker` is what to draw in the gutter: "•" for
        /// bulleted lists, "1." and friends for numbered ones.
        case bullet(marker: String)
        case quote
        case rule
        case code(language: String?)
    }

    let id: Int
    let kind: Kind
    let text: String
}

struct ParsedMessage: Equatable {
    /// Everything the model wrote inside think tags, concatenated.
    var reasoning: String = ""
    /// True while a think block is still open — i.e. the model is reasoning
    /// right now and the answer has not started.
    var reasoningIsOpen: Bool = false
    /// Everything outside the think tags, verbatim. This is what goes back
    /// into a re-hydrated chat history — rejoining the parsed blocks would
    /// drop code fences and list markers.
    var answer: String = ""
    var blocks: [MessageBlock] = []

    var hasReasoning: Bool {
        reasoningIsOpen || !reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var hasAnswer: Bool { !blocks.isEmpty }

    static func parse(_ source: String) -> ParsedMessage {
        let split = splitReasoning(source)
        return ParsedMessage(
            reasoning: split.reasoning,
            reasoningIsOpen: split.isOpen,
            answer: split.answer,
            blocks: parseBlocks(split.answer)
        )
    }

    // MARK: Reasoning

    /// Ling 3.0 and Qwen3.5 both fence their chain of thought, but not with the
    /// same tag name, so both spellings are recognised.
    private static let thinkTags = [("<think>", "</think>"), ("<thinking>", "</thinking>")]

    private static func splitReasoning(
        _ text: String
    ) -> (reasoning: String, answer: String, isOpen: Bool) {
        var reasoning = ""
        var answer = ""
        var isOpen = false
        var cursor = text.startIndex

        while cursor < text.endIndex {
            // Whichever opening tag comes first from here on.
            var opening: (range: Range<String.Index>, close: String)?
            for (open, close) in thinkTags {
                guard let range = text.range(of: open, range: cursor..<text.endIndex) else {
                    continue
                }
                if opening == nil || range.lowerBound < opening!.range.lowerBound {
                    opening = (range, close)
                }
            }

            guard let opening else {
                answer += text[cursor...]
                break
            }

            answer += text[cursor..<opening.range.lowerBound]

            if let closing = text.range(
                of: opening.close, range: opening.range.upperBound..<text.endIndex)
            {
                reasoning += text[opening.range.upperBound..<closing.lowerBound]
                cursor = closing.upperBound
            } else {
                // Still streaming inside the block.
                reasoning += text[opening.range.upperBound...]
                isOpen = true
                break
            }
        }

        return (reasoning, answer, isOpen)
    }

    // MARK: Blocks

    private static func parseBlocks(_ text: String) -> [MessageBlock] {
        var blocks: [MessageBlock] = []
        var paragraph: [String] = []
        var next = 0

        func emit(_ kind: MessageBlock.Kind, _ body: String) {
            blocks.append(MessageBlock(id: next, kind: kind, text: body))
            next += 1
        }

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            let body = paragraph.joined(separator: "\n")
            paragraph.removeAll()
            guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
            emit(.paragraph, body)
        }

        var lines = text.components(separatedBy: .newlines)[...]

        while let line = lines.first {
            lines = lines.dropFirst()
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                flushParagraph()
                let fence = String(trimmed.prefix(3))
                let language = trimmed.dropFirst(3).trimmingCharacters(in: .whitespaces)
                var body: [String] = []
                while let codeLine = lines.first {
                    lines = lines.dropFirst()
                    if codeLine.trimmingCharacters(in: .whitespaces).hasPrefix(fence) { break }
                    body.append(codeLine)
                }
                emit(
                    .code(language: language.isEmpty ? nil : language),
                    body.joined(separator: "\n"))
                continue
            }

            if trimmed.isEmpty {
                flushParagraph()
                continue
            }

            if isRule(trimmed) {
                flushParagraph()
                emit(.rule, "")
                continue
            }

            if trimmed.hasPrefix("#") {
                let hashes = trimmed.prefix { $0 == "#" }.count
                let rest = trimmed.dropFirst(hashes).trimmingCharacters(in: .whitespaces)
                // Markdown wants a space after the hashes; without that check a
                // line like "#1 sırada" becomes a heading.
                if hashes <= 6, trimmed.dropFirst(hashes).first == " ", !rest.isEmpty {
                    flushParagraph()
                    emit(.heading(level: min(hashes, 3)), rest)
                    continue
                }
            }

            if trimmed.hasPrefix("> ") || trimmed == ">" {
                flushParagraph()
                emit(.quote, String(trimmed.dropFirst(1)).trimmingCharacters(in: .whitespaces))
                continue
            }

            if let marker = listMarker(trimmed) {
                flushParagraph()
                emit(
                    .bullet(marker: marker.rendered),
                    String(trimmed.dropFirst(marker.length))
                        .trimmingCharacters(in: .whitespaces))
                continue
            }

            paragraph.append(line)
        }

        flushParagraph()
        return blocks
    }

    private static func isRule(_ trimmed: String) -> Bool {
        guard trimmed.count >= 3 else { return false }
        let unique = Set(trimmed)
        return unique.count == 1 && (unique.first == "-" || unique.first == "*"
            || unique.first == "_")
    }

    /// Recognises `- `, `* `, `+ ` and `12. ` / `12) ` list openers.
    private static func listMarker(_ trimmed: String) -> (rendered: String, length: Int)? {
        if let first = trimmed.first, "-*+".contains(first),
            trimmed.dropFirst().first == " "
        {
            return ("•", 2)
        }

        let digits = trimmed.prefix { $0.isNumber }
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let after = trimmed.dropFirst(digits.count)
        guard let punctuation = after.first, punctuation == "." || punctuation == ")",
            after.dropFirst().first == " "
        else { return nil }
        return ("\(digits).", digits.count + 2)
    }
}

enum InlineMarkdown {
    /// Inline-only so that a hard-wrapped paragraph keeps its line breaks —
    /// block structure is handled by the parser above, not by Foundation.
    static func attributed(_ source: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: false,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible)
        return (try? AttributedString(markdown: source, options: options))
            ?? AttributedString(source)
    }
}
