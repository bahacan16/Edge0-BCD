// Reading a DXF well enough to talk about it.
//
// A DXF is text, so the naive move is to hand the model the file. That fails on
// arithmetic: the six-entity drawing used to develop this is 37 KB, and a real
// architectural plan is megabytes. Forty layers of streamed experts times ten
// thousand tokens of group codes is minutes of prompt for a question about
// three layers.
//
// It is also the wrong text. DXF is a flat stream of (code, value) pairs where
// the structure lives in the codes, not in the layout — a model reading it raw
// spends its attention reconstructing what a parser can state in one line.
//
// So the file is parsed and summarised: version, units, extents, entity counts
// by type, layers with their colours and how much is actually drawn on each,
// and the block definitions. That is what someone asking "which layer is the
// wall on" or "why is this drawing empty" needs, in a few hundred tokens.
//
// The algorithm was written in Python first and checked against ezdxf reading
// the same files, because being subtly wrong about someone's drawing is worse
// than refusing it. It agrees with ezdxf on layers, blocks, entity counts and
// extents. Writing it in Swift also removes the reason to embed a Python
// runtime just to answer "what is in this file".

import Foundation

enum Edge0DXF {

    /// `$ACADVER` codes, which are how a DXF states its version.
    private static let releases = [
        "AC1006": "R10", "AC1009": "R12", "AC1012": "R13", "AC1014": "R14",
        "AC1015": "R2000", "AC1018": "R2004", "AC1021": "R2007", "AC1024": "R2010",
        "AC1027": "R2013", "AC1032": "R2018",
    ]

    /// `$INSUNITS`.
    private static let units = [
        0: "birimsiz", 1: "inç", 2: "feet", 3: "mil", 4: "mm", 5: "cm", 6: "m",
        9: "mikron", 10: "yard", 11: "ångström", 14: "desimetre",
    ]

    /// True for something that opens like a DXF. Cheap enough to run on the
    /// first few hundred bytes.
    static func looksLikeDXF(_ text: String) -> Bool {
        let head = text.prefix(4096)
        return head.contains("SECTION") && (head.contains("HEADER") || head.contains("$ACADVER"))
    }

    struct Drawing {
        var release = "?"
        var insertionUnits: Int?
        var layers: [String: (color: Int, linetype: String)] = [:]
        var blocks: Set<String> = []
        var entityCounts: [String: Int] = [:]
        var entitiesPerLayer: [String: Int] = [:]
        var minX = Double.infinity
        var maxX = -Double.infinity
        var minY = Double.infinity
        var maxY = -Double.infinity

        var entityTotal: Int { entityCounts.values.reduce(0, +) }
        var hasExtents: Bool { minX <= maxX && minY <= maxY }
    }

    // MARK: Parse

    static func parse(_ text: String) -> Drawing {
        var drawing = Drawing()
        var section = ""
        var entry = ""          // the current 0-code entry inside a section
        var headerKey = ""
        var headerValues = 0
        var layerBeingRead: String?

        // A DXF is strictly alternating lines: a group code, then its value.
        // Anything else is a malformed file, and the skip below is what keeps
        // one stray line from shifting every pair after it.
        var code: Int?
        text.enumerateLines { line, _ in
            let token = line.trimmingCharacters(in: .whitespaces)
            guard let pending = code else {
                code = Int(token)
                return
            }
            code = nil
            handle(pending, token, &drawing, &section, &entry, &headerKey,
                &headerValues, &layerBeingRead)
        }
        return drawing
    }

    private static func handle(
        _ code: Int, _ value: String, _ drawing: inout Drawing, _ section: inout String,
        _ entry: inout String, _ headerKey: inout String, _ headerValues: inout Int,
        _ layerBeingRead: inout String?
    ) {
        if code == 0 {
            switch value {
            case "SECTION":
                section = "?"
            case "ENDSEC":
                section = ""
                entry = ""
            default:
                switch section {
                case "TABLES":
                    // Every 0-code in TABLES names an entry type, so STYLE and
                    // APPID reset it too. Tracking only LAYER counted every
                    // other table's entries as layers — fifty-five of them on a
                    // drawing that has four.
                    entry = value
                    layerBeingRead = nil
                case "BLOCKS":
                    entry = value
                case "ENTITIES":
                    entry = value
                    drawing.entityCounts[value, default: 0] += 1
                default:
                    break
                }
            }
            return
        }

        if section == "?" {
            if code == 2 { section = value }
            return
        }

        switch section {
        case "HEADER":
            if code == 9 {
                headerKey = value
                headerValues = 0
            } else if !headerKey.isEmpty {
                if headerKey == "$ACADVER", code == 1 { drawing.release = value }
                if headerKey == "$INSUNITS", code == 70 { drawing.insertionUnits = Int(value) }
                headerValues += 1
                if headerValues >= 3 { headerKey = "" }
            }

        case "TABLES":
            guard entry == "LAYER" else { return }
            if code == 2 {
                layerBeingRead = value
                if drawing.layers[value] == nil {
                    drawing.layers[value] = (color: 256, linetype: "CONTINUOUS")
                }
            } else if let name = layerBeingRead, var layer = drawing.layers[name] {
                if code == 62 { layer.color = Int(value) ?? layer.color }
                if code == 6 { layer.linetype = value }
                drawing.layers[name] = layer
            }

        case "BLOCKS":
            if code == 2, entry == "BLOCK" { drawing.blocks.insert(value) }

        case "ENTITIES":
            if code == 8 {
                drawing.entitiesPerLayer[value, default: 0] += 1
            } else if (10 ... 13).contains(code), let x = Double(value) {
                drawing.minX = min(drawing.minX, x)
                drawing.maxX = max(drawing.maxX, x)
            } else if (20 ... 23).contains(code), let y = Double(value) {
                drawing.minY = min(drawing.minY, y)
                drawing.maxY = max(drawing.maxY, y)
            }

        default:
            break
        }
    }

    // MARK: Summary

    static func summary(of text: String, name: String) -> String {
        let drawing = parse(text)
        var lines = ["DXF çizimi: \(name)"]
        lines.append("sürüm: \(releases[drawing.release] ?? drawing.release)")
        if let code = drawing.insertionUnits {
            lines.append("birim: \(units[code] ?? "kod \(code)")")
        }
        if drawing.hasExtents {
            lines.append(
                String(
                    format: "sınırlar: X %.6g .. %.6g · Y %.6g .. %.6g",
                    drawing.minX, drawing.maxX, drawing.minY, drawing.maxY))
        }
        lines.append("toplam varlık: \(drawing.entityTotal)")

        if !drawing.entityCounts.isEmpty {
            let types =
                drawing.entityCounts
                .sorted { ($0.value, $1.key) > ($1.value, $0.key) }
                .map { "\($0.key) × \($0.value)" }
                .joined(separator: ", ")
            lines.append("varlık türleri: \(types)")
        }

        if !drawing.layers.isEmpty {
            lines.append("katmanlar (\(drawing.layers.count)):")
            for (name, layer) in drawing.layers.sorted(by: { $0.key < $1.key }) {
                let drawn = drawing.entitiesPerLayer[name] ?? 0
                lines.append(
                    "  \(name) · renk \(layer.color) · \(layer.linetype)"
                        + (drawn > 0 ? " · \(drawn) varlık" : " · boş"))
            }
        }

        // The automatic ones are noise: every DXF has *Model_Space and
        // *Paper_Space, and nobody asking about a drawing means those.
        let named = drawing.blocks.filter { !$0.hasPrefix("*") }.sorted()
        if !named.isEmpty {
            lines.append("bloklar (\(named.count)): \(named.joined(separator: ", "))")
        }

        // Entities referencing a layer the table never declared are a real
        // fault and a common one — worth saying, since it is exactly the sort
        // of thing someone opens a drawing to find out.
        let undeclared = drawing.entitiesPerLayer.keys.filter { drawing.layers[$0] == nil }
        if !undeclared.isEmpty {
            lines.append(
                "UYARI: tabloda tanımlı olmayan katmanlarda varlık var: "
                    + undeclared.sorted().joined(separator: ", "))
        }
        return lines.joined(separator: "\n")
    }
}
