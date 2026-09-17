// Getting the text out of a PDF.
//
// Two extractors, tried in that order. PDFKit is part of iOS, costs nothing to
// call and handles the ordinary case — a PDF exported from Word, a datasheet, a
// scanned-then-OCR'd document — including its font encodings.
//
// pypdf is the fallback, and it earns its place on the files PDFKit shrugs at:
// generator-specific quirks where PDFKit returns an empty string while pypdf's
// content-stream walk still finds the words. It costs an interpreter start, so
// it is not the first thing tried, only the second.
//
// Neither reads a scan. A PDF that is nothing but page images has no text in
// it, and saying so plainly is better than handing the model an empty file and
// letting it invent the contents.

import Foundation
import PDFKit

enum Edge0PDF {
    struct Extract {
        var text: String
        var pageCount: Int
        /// Which extractor produced the text, so the chip and the prompt can
        /// say, and so a bad extraction can be told from a bad file.
        var source: String
        var truncated: Bool
    }

    /// Below this, a document is treated as having no extractable text at all
    /// rather than as a short one — a stray page number is not content.
    private static let meaningfulCharacters = 24

    static func extract(_ data: Data, name: String, limit: Int) -> Extract? {
        guard let document = PDFDocument(data: data) else { return nil }
        if document.isLocked {
            // An unlocked-by-empty-password PDF is common enough to try.
            _ = document.unlock(withPassword: "")
        }
        let native = pdfKitText(document, limit: limit)
        if native.text.trimmingCharacters(in: .whitespacesAndNewlines).count >= meaningfulCharacters
        {
            return native
        }
        if let viaPython = pypdfText(data, pageCount: document.pageCount, limit: limit),
            viaPython.text.trimmingCharacters(in: .whitespacesAndNewlines).count
                >= meaningfulCharacters
        {
            return viaPython
        }
        // Keep the native result even when it is empty: the page count is still
        // true, and the caller reports "no text" from it.
        return native
    }

    private static func pdfKitText(_ document: PDFDocument, limit: Int) -> Extract {
        var pieces: [String] = []
        var bytes = 0
        var truncated = false
        for index in 0..<document.pageCount {
            guard let page = document.page(at: index), let body = page.string else { continue }
            let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let block = "--- sayfa \(index + 1) ---\n\(trimmed)"
            bytes += block.utf8.count
            if bytes > limit {
                truncated = true
                break
            }
            pieces.append(block)
        }
        return Extract(
            text: pieces.joined(separator: "\n\n"), pageCount: document.pageCount,
            source: "PDFKit", truncated: truncated)
    }

    /// The same job through pypdf, for the files PDFKit gives up on.
    private static func pypdfText(_ data: Data, pageCount: Int, limit: Int) -> Extract? {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("pdf-\(UUID().uuidString).pdf")
        guard (try? data.write(to: scratch)) != nil else { return nil }
        defer { try? FileManager.default.removeItem(at: scratch) }

        let script = """
            import json
            from pypdf import PdfReader
            reader = PdfReader(\(Edge0Python.literal(scratch.path)))
            if reader.is_encrypted:
                reader.decrypt("")
            limit = \(limit)
            pieces = []
            size = 0
            cut = False
            for index, page in enumerate(reader.pages):
                body = (page.extract_text() or "").strip()
                if not body:
                    continue
                block = "--- sayfa %d ---\\n%s" % (index + 1, body)
                size += len(block.encode("utf-8"))
                if size > limit:
                    cut = True
                    break
                pieces.append(block)
            _edge0_write(json.dumps({
                "pages": len(reader.pages),
                "text": "\\n\\n".join(pieces),
                "truncated": cut,
            }))
            """
        guard let output = try? Edge0Python.run(script), !output.isEmpty,
            !output.hasPrefix("HATA"),
            let json = output.data(using: .utf8),
            let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
            let text = root["text"] as? String
        else {
            Edge0Log.write("pypdf metin çıkaramadı")
            return nil
        }
        Edge0Log.write("pdf metni pypdf ile çıkarıldı")
        return Extract(
            text: text, pageCount: (root["pages"] as? Int) ?? pageCount, source: "pypdf",
            truncated: (root["truncated"] as? Bool) ?? false)
    }
}
