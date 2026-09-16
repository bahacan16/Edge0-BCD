// Parsing a tool call out of what the model actually wrote.
//
// Measured on the device, both tiers call tools and neither writes the format
// everyone assumes. The textbook shape is JSON inside a wrapper —
// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` — and it is what a
// parser written from memory would look for. What these checkpoints write:
//
//   edge0-8b (Ling):
//     <tool_call>get_weather
//     <arg_key>city</arg_key>
//     <arg_value>İstanbul</arg_value>
//     </tool_call>
//
//   edge0-35b (Qwen3.5):
//     <tool_call>
//     <function=get_weather>
//     <parameter=city>
//     İstanbul
//     </parameter>
//     </function>
//     </tool_call>
//
// Both XML-ish, both different from each other, and a parser for either one
// reading the other sees prose. So all three dialects are handled, and which
// one a checkpoint speaks is discovered rather than configured — a tier's
// format is a property of its template, and the next checkpoint edge0 ships
// may well speak a fourth.

import Foundation

struct Edge0ToolCall: Equatable, Sendable {
    let name: String
    /// Everything arrives as text. These templates carry no types — the model
    /// writes `42` and `"42"` identically — so a caller that wants a number
    /// converts one, rather than this pretending to know which was meant.
    let arguments: [String: String]
    /// Which shape it was written in, for the log and for the probe.
    let dialect: Dialect

    enum Dialect: String, Sendable {
        case lingArgKey = "Ling (arg_key/arg_value)"
        case qwenFunction = "Qwen (function=/parameter=)"
        case json = "JSON"
    }
}

enum Edge0ToolCallParser {
    private static let open = "<tool_call>"
    private static let close = "</tool_call>"

    /// Every call in `text`, in the order written.
    ///
    /// A truncated block still parses: generation stops on a token budget, and
    /// a call cut off after its last argument is complete enough to run. The
    /// closing tag is a convenience, not a requirement.
    static func calls(in text: String) -> [Edge0ToolCall] {
        var results: [Edge0ToolCall] = []
        var rest = Substring(text)

        while let start = rest.range(of: open) {
            let after = rest[start.upperBound...]
            let body: Substring
            if let end = after.range(of: close) {
                body = after[..<end.lowerBound]
                rest = after[end.upperBound...]
            } else {
                body = after
                rest = after[after.endIndex...]
            }
            if let call = parse(body) { results.append(call) }
        }
        return results
    }

    private static func parse(_ body: Substring) -> Edge0ToolCall? {
        if let call = parseQwen(body) { return call }
        if let call = parseLing(body) { return call }
        return parseJSON(body)
    }

    // MARK: Dialects

    /// `<function=name>` with `<parameter=key>value</parameter>` inside.
    private static func parseQwen(_ body: Substring) -> Edge0ToolCall? {
        guard let name = tagValue(in: body, prefix: "<function=") else { return nil }
        var arguments: [String: String] = [:]
        var rest = body
        while let open = rest.range(of: "<parameter=") {
            let after = rest[open.upperBound...]
            guard let nameEnd = after.firstIndex(of: ">") else { break }
            let key = String(after[..<nameEnd])
            let valueStart = after.index(after: nameEnd)
            let remainder = after[valueStart...]
            let value: Substring
            if let end = remainder.range(of: "</parameter>") {
                value = remainder[..<end.lowerBound]
                rest = remainder[end.upperBound...]
            } else {
                // Truncated, or a template that closes with the next tag
                // rather than a matching one.
                let stop =
                    remainder.range(of: "<parameter=")?.lowerBound
                    ?? remainder.range(of: "</function>")?.lowerBound
                    ?? remainder.endIndex
                value = remainder[..<stop]
                rest = remainder[stop...]
            }
            arguments[key] = trimmed(value)
        }
        return Edge0ToolCall(name: name, arguments: arguments, dialect: .qwenFunction)
    }

    /// The name on the opening line, then `<arg_key>`/`<arg_value>` pairs.
    private static func parseLing(_ body: Substring) -> Edge0ToolCall? {
        guard body.contains("<arg_key>") else { return nil }
        guard let firstLine = body.split(separator: "\n", maxSplits: 1).first else { return nil }
        let name = trimmed(firstLine)
        guard !name.isEmpty, !name.hasPrefix("<") else { return nil }

        // Zipped in order rather than matched pairwise: the template emits them
        // strictly alternating, and a key whose value is missing (truncation)
        // is better dropped than paired with the next key's value.
        let keys = values(in: body, tag: "arg_key")
        let vals = values(in: body, tag: "arg_value")
        var arguments: [String: String] = [:]
        for (key, value) in zip(keys, vals) { arguments[key] = value }
        return Edge0ToolCall(name: name, arguments: arguments, dialect: .lingArgKey)
    }

    /// `{"name": ..., "arguments": {...}}`, the shape most models write.
    private static func parseJSON(_ body: Substring) -> Edge0ToolCall? {
        guard let start = body.firstIndex(of: "{"), let end = body.lastIndex(of: "}"),
            start < end
        else { return nil }
        let json = String(body[start ... end])
        guard let data = json.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let name = object["name"] as? String
        else { return nil }

        var arguments: [String: String] = [:]
        if let raw = object["arguments"] ?? object["parameters"] {
            if let dictionary = raw as? [String: Any] {
                for (key, value) in dictionary { arguments[key] = describe(value) }
            } else if let text = raw as? String,
                let nested = text.data(using: .utf8),
                let dictionary = try? JSONSerialization.jsonObject(with: nested)
                    as? [String: Any]
            {
                // Some models write the arguments as a JSON *string*.
                for (key, value) in dictionary { arguments[key] = describe(value) }
            }
        }
        return Edge0ToolCall(name: name, arguments: arguments, dialect: .json)
    }

    // MARK: Bits

    private static func tagValue(in body: Substring, prefix: String) -> String? {
        guard let open = body.range(of: prefix) else { return nil }
        let after = body[open.upperBound...]
        guard let end = after.firstIndex(of: ">") else { return nil }
        let name = trimmed(after[..<end])
        return name.isEmpty ? nil : name
    }

    private static func values(in body: Substring, tag: String) -> [String] {
        var results: [String] = []
        var rest = body
        while let open = rest.range(of: "<\(tag)>") {
            let after = rest[open.upperBound...]
            guard let end = after.range(of: "</\(tag)>") else { break }
            results.append(trimmed(after[..<end.lowerBound]))
            rest = after[end.upperBound...]
        }
        return results
    }

    private static func describe(_ value: Any) -> String {
        switch value {
        case let text as String: text
        case let number as NSNumber: number.stringValue
        default:
            (try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "\(value)"
        }
    }

    private static func trimmed(_ text: Substring) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
