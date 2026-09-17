// Text files attached to a turn.
//
// Everything that reaches the model reaches it as text, so every kind of file
// here is turned into text first: a .py or a .txt goes in as it stands, a .dxf
// goes in as a description of itself, a PDF goes in as the words extracted from
// it. What is never done is handing over raw bytes of a container format and
// hoping — that produces confident nonsense, which is worse than refusing.
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
        /// Text pulled out of a container, which is not the file's own bytes.
        case pdf
    }

    let id = UUID()
    let name: String
    let text: String
    /// Bytes on disk, which is not `text.count` once a file has been truncated
    /// or, for a drawing, summarised.
    let byteCount: Int
    let truncated: Bool
    var kind: Kind = .text
    /// What was done to the file to make it readable, for the kinds where that
    /// is not obvious: "12 sayfa · PDFKit".
    var descriptor: String? = nil

    var summary: String {
        let size = ByteCountFormatter.string(fromByteCount: Int64(byteCount), countStyle: .file)
        if kind == .drawing { return "\(size) · çizim özeti" }
        if kind == .pdf, let descriptor {
            return truncated ? "\(size) · \(descriptor) · kırpıldı" : "\(size) · \(descriptor)"
        }
        return truncated ? "\(size) · kırpıldı" : size
    }
}

enum Edge0AttachmentError: LocalizedError {
    case unreadable(String)
    case notText(String)
    case empty(String)
    case dwgUnsupported(String)
    case pdfUnreadable(String)
    case pdfWithoutText(String)

    var errorDescription: String? {
        switch self {
        case .unreadable(let name): "Dosya okunamadı: \(name)"
        case .notText(let name):
            "\(name) ikili bir dosya, metin değil. Görseller, arşivler ve"
                + " çalıştırılabilir dosyalar modele verilemez — metin dosyaları"
                + " (.txt, .md, .json, .csv), kaynak kodu, .pdf ve .dxf eklenebilir."
        case .pdfUnreadable(let name):
            "\(name) açılamadı — bozuk ya da parola korumalı bir PDF."
        case .pdfWithoutText(let name):
            "\(name) içinde metin yok; sayfaları görüntü olan taranmış bir PDF"
                + " gibi duruyor. Uygulamada metin tanıma (OCR) yok, o yüzden"
                + " okunacak bir şey çıkmıyor."
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

    /// What the picker will offer: every regular file, and no folders.
    ///
    /// This used to be a curated list of text UTIs plus
    /// `UTType(filenameExtension: "dxf")` and the same for `dwg`. Neither
    /// extension has a registered system type, so those two calls return
    /// *dynamic* UTIs — `dyn.ah62d4rv4ge81k5pu` and the like — and a document
    /// picker handed a dynamic type does not reliably return the file it was
    /// given. DXF is the format this is most needed for, so the list was
    /// disqualifying its most important case.
    ///
    /// `.data` is every file and no directory, which keeps a folder from being
    /// picked as if it were a document. What the app can actually read is
    /// decided by `read` a moment later, and it says why when the answer is
    /// no — which was always the intent here: a file greyed out in the picker
    /// with no explanation is worse than one that says why.
    static let contentTypes: [UTType] = [.data]

    /// A drawing is summarised rather than quoted, so it may be far larger than
    /// a file whose text goes into the prompt as it stands.
    static let maximumDrawingBytes = 48 * 1024 * 1024

    static func read(_ url: URL) throws -> Edge0Attachment {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        let name = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        Edge0Log.write("ek okunuyor: \(name) · kapsam \(scoped)")
        guard ext != "dwg" else { throw Edge0AttachmentError.dwgUnsupported(name) }

        guard let data = coordinatedRead(url) else {
            throw Edge0AttachmentError.unreadable(name)
        }
        guard !data.isEmpty else { throw Edge0AttachmentError.empty(name) }

        if ext == "dxf" { return try readDrawing(data, name: name) }
        if ext == "pdf" || data.starts(with: Array("%PDF".utf8)) {
            return try readPDF(data, name: name)
        }

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

    /// A PDF goes in as its words, with page markers kept so the model can say
    /// "on page 4" and mean it.
    private static func readPDF(_ data: Data, name: String) throws -> Edge0Attachment {
        guard data.count <= maximumDrawingBytes else {
            throw Edge0AttachmentError.unreadable(name)
        }
        guard let extract = Edge0PDF.extract(data, name: name, limit: maximumBytes) else {
            throw Edge0AttachmentError.pdfUnreadable(name)
        }
        guard !extract.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Edge0AttachmentError.pdfWithoutText(name)
        }
        return Edge0Attachment(
            name: name, text: extract.text, byteCount: data.count, truncated: extract.truncated,
            kind: .pdf, descriptor: "\(extract.pageCount) sayfa · \(extract.source)")
    }

    /// Reads through `NSFileCoordinator`.
    ///
    /// A file picked out of the Files app may live in iCloud Drive or in
    /// another app's provider and not be on this device yet. A bare
    /// `Data(contentsOf:)` on one of those fails, or blocks — coordinating the
    /// read is what makes the provider materialise the file first.
    private static func coordinatedRead(_ url: URL) -> Data? {
        var data: Data?
        var failure: Error?
        var coordinationError: NSError?
        NSFileCoordinator().coordinate(
            readingItemAt: url, options: [.withoutChanges], error: &coordinationError
        ) { actual in
            do { data = try Data(contentsOf: actual) } catch { failure = error }
        }
        if let coordinationError {
            Edge0Log.failure("ek eşgüdümlü okuma", coordinationError)
            // The coordinator refusing is not the last word: a plain read still
            // works for a file already on the device.
            return try? Data(contentsOf: url)
        }
        if let failure { Edge0Log.failure("ek okuma", failure) }
        return data
    }

    /// Whether these bytes are a file of text at all.
    ///
    /// This has to be asked *before* decoding, because decoding cannot answer
    /// it. ISO-Latin-1 maps every one of the 256 byte values to a character,
    /// so `String(data:encoding:.isoLatin1)` succeeds on anything — and with
    /// that in the fallback list, `decode` never returned nil and `notText`
    /// could never be thrown. A PNG went into a prompt as sixty kilobytes of
    /// mojibake, which is tens of thousands of tokens of noise, and the turn
    /// that followed took the app out of memory.
    ///
    /// Two signals, both cheap: a NUL byte, which no text file has, and the
    /// share of control characters, which in real text is approximately zero
    /// and in compressed binary is about a sixth.
    private static func looksBinary(_ data: Data) -> Bool {
        // UTF-16 text is half NUL bytes by construction, so it has to be
        // recognised before the NUL test rather than after it.
        if hasUTF16BOM(data) { return false }
        let sample = data.prefix(8 * 1024)
        guard !sample.isEmpty else { return false }
        var controls = 0
        for byte in sample {
            if byte == 0 { return true }
            // Everything below space except tab, newline and carriage return,
            // plus the delete character.
            if (byte < 0x09) || (byte > 0x0D && byte < 0x20) || byte == 0x7F {
                controls += 1
            }
        }
        return controls * 100 > sample.count * 2
    }

    private static func hasUTF16BOM(_ data: Data) -> Bool {
        let head = Array(data.prefix(2))
        return head == [0xFF, 0xFE] || head == [0xFE, 0xFF]
    }

    private static func decode(_ data: Data) -> String? {
        guard !looksBinary(data) else { return nil }
        if hasUTF16BOM(data), let utf16 = String(data: data, encoding: .utf16) { return utf16 }
        if let utf8 = String(data: data, encoding: .utf8) { return utf8 }
        // Latin-1 is last and is a guess, not a test: it accepts anything that
        // reaches it. What keeps that honest is the screening above.
        for encoding in [String.Encoding.utf16, .windowsCP1254, .isoLatin1] {
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
            var title = "Dosya: \(attachment.name)"
            if attachment.kind == .pdf, let descriptor = attachment.descriptor {
                title += " (\(descriptor) ile çıkarılan metin)"
            }
            var block = "\(title)\n\(fence)\n\(attachment.text)\n\(fence)"
            if attachment.truncated {
                block += "\n(Dosyanın ilk \(Edge0AttachmentReader.maximumBytes / 1024) KB'ı.)"
            }
            return block
        }
        .joined(separator: "\n\n")
    }
}
