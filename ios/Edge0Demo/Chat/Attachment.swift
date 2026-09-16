// Text files attached to a turn.
//
// Deliberately text only. A phone-sized model reads what it is given as tokens,
// so a file is useful here exactly to the extent that it is readable — a .py or
// a .txt or a .json is a prompt with a name on it, while a PDF or a .docx is a
// container that would need parsing this app does not do. Handing the model the
// raw bytes of one of those produces confident nonsense, which is worse than
// refusing it.
//
// Size is capped because the cost is not storage, it is the prompt. On the
// streaming tier every thousand tokens of context is another pass over forty
// layers of experts read from disk, and a 2 MB log file pasted into a turn
// would take minutes before the first word of the answer.

import Foundation
import UniformTypeIdentifiers

struct Edge0Attachment: Identifiable, Hashable, Sendable {
    enum Kind: Sendable {
        /// The file's own text goes into the prompt.
        case text
        /// A description of the file goes in, because the file itself is too
        /// large and in the wrong shape to be read directly.
        case drawing
    }

    let id = UUID()
    let name: String
    let text: String
    /// Bytes on disk, which is not `text.count` once a file has been truncated
    /// or, for a drawing, summarised.
    let byteCount: Int
    let truncated: Bool
    var kind: Kind = .text

    var summary: String {
        let size = ByteCountFormatter.string(fromByteCount: Int64(byteCount), countStyle: .file)
        if kind == .drawing { return "\(size) · çizim özeti" }
        return truncated ? "\(size) · kırpıldı" : size
    }
}

enum Edge0AttachmentError: LocalizedError {
    case unreadable(String)
    case notText(String)
    case empty(String)
    case dwgUnsupported(String)

    var errorDescription: String? {
        switch self {
        case .unreadable(let name): "Dosya okunamadı: \(name)"
        case .notText(let name):
            "\(name) düz metin değil. Metin dosyaları (.txt, .md, .json, .csv), kaynak kodu ve .dxf eklenebilir."
        case .empty(let name): "\(name) boş."
        case .dwgUnsupported(let name):
            """
            \(name) DWG. DWG kapalı, belgelenmemiş bir ikili biçim — ezdxf dahil             hiçbir açık kütüphane okuyamıyor, Autodesk'in kendi dönüştürücüsü de             iOS'ta çalışmıyor. CAD programınızdan DXF olarak dışa aktarın.
            """
        }
    }
}

enum Edge0AttachmentReader {
    /// Per file. Roughly fifteen thousand tokens, which is already a slow
    /// prompt on the streaming tier.
    static let maximumBytes = 60 * 1024

    /// What the picker will offer. `.item` is deliberately absent: everything
    /// here is something whose bytes are text.
    static let contentTypes: [UTType] = [
        .plainText, .utf8PlainText, .text, .sourceCode, .pythonScript, .swiftSource,
        .cSource, .cHeader, .json, .xml, .yaml, .commaSeparatedText, .tabSeparatedText,
        .html, .log, .rtf, .delimitedText, .script, .shellScript, .propertyList,
    ] + [
        // Neither has a system type, so they are declared by extension. DWG is
        // offered on purpose even though it cannot be read: a file greyed out
        // in the picker with no explanation is worse than one that says why.
        UTType(filenameExtension: "dxf"), UTType(filenameExtension: "dwg"),
    ].compactMap { $0 }

    /// A drawing is summarised rather than quoted, so it may be far larger than
    /// a file whose text goes into the prompt as it stands.
    static let maximumDrawingBytes = 48 * 1024 * 1024

    static func read(_ url: URL) throws -> Edge0Attachment {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        let name = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        guard ext != "dwg" else { throw Edge0AttachmentError.dwgUnsupported(name) }

        guard let data = try? Data(contentsOf: url) else {
            throw Edge0AttachmentError.unreadable(name)
        }
        guard !data.isEmpty else { throw Edge0AttachmentError.empty(name) }

        if ext == "dxf" { return try readDrawing(data, name: name) }

        // Decoded rather than sniffed by extension: a .log written by a Windows
        // tool is as likely to be UTF-16 as UTF-8, and a file that decodes is
        // text whatever it is called.
        let head = data.prefix(maximumBytes)
        guard let text = decode(head) else { throw Edge0AttachmentError.notText(name) }

        // A truncation in the middle of a multi-byte character would be decoded
        // away above, but one in the middle of a *line* is just confusing, so
        // the tail is trimmed back to the last newline when there is one.
        var body = text
        let truncated = data.count > maximumBytes
        if truncated, let lastBreak = body.lastIndex(of: "\n") {
            body = String(body[..<lastBreak])
        }

        return Edge0Attachment(
            name: name, text: body, byteCount: data.count, truncated: truncated)
    }

    /// A drawing goes in as a description of itself.
    ///
    /// Quoting one is not an option: a six-entity test drawing is 37 KB of
    /// group codes and a real plan is megabytes, which on the streaming tier is
    /// minutes of prompt before a word of the answer. The summary answers the
    /// questions a drawing gets asked — what is on which layer, how big is it,
    /// what blocks does it use — in a few hundred tokens.
    private static func readDrawing(_ data: Data, name: String) throws -> Edge0Attachment {
        guard data.count <= maximumDrawingBytes else {
            throw Edge0AttachmentError.unreadable(name)
        }
        guard let text = decode(data) else { throw Edge0AttachmentError.notText(name) }
        guard Edge0DXF.looksLikeDXF(text) else {
            // Binary DXF exists, is rare, and is not this.
            throw Edge0AttachmentError.notText(name)
        }
        return Edge0Attachment(
            name: name, text: Edge0DXF.summary(of: text, name: name),
            byteCount: data.count, truncated: false, kind: .drawing)
    }

    private static func decode(_ data: Data) -> String? {
        if let utf8 = String(data: data, encoding: .utf8) { return utf8 }
        for encoding in [String.Encoding.utf16, .isoLatin1, .windowsCP1254] {
            if let text = String(data: data, encoding: encoding) { return text }
        }
        return nil
    }
}

extension Array where Element == Edge0Attachment {
    /// The files as one block ahead of the question, each fenced and named.
    ///
    /// The name matters as much as the contents: "fix the bug" means something
    /// different about `main.py` than about `notes.txt`, and the model has no
    /// other way to know which it is holding.
    func promptBlock() -> String {
        guard !isEmpty else { return "" }
        return map { attachment in
            if attachment.kind == .drawing {
                // No fence: this is already prose about a file rather than the
                // file, and fencing it would invite the model to treat it as
                // something to quote back.
                return attachment.text
            }
            let fence = attachment.text.contains("```") ? "````" : "```"
            var block = "Dosya: \(attachment.name)\n\(fence)\n\(attachment.text)\n\(fence)"
            if attachment.truncated {
                block += "\n(Dosyanın ilk \(Edge0AttachmentReader.maximumBytes / 1024) KB'ı.)"
            }
            return block
        }
        .joined(separator: "\n\n")
    }
}
