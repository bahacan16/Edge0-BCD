// Installing and removing Python packages, without pip.
//
// pip itself is not a good fit here: it wants to compile, it wants a writable
// prefix, and it is a large moving part to debug through a phone screen. But
// the part of it that matters on iOS is small. A pure-Python wheel is a zip
// with a manifest, PyPI publishes an index that says which wheels are pure,
// and a package is "installed" once its files sit on `sys.path`. So the whole
// job is: ask the index, download, unzip, read the dependencies, repeat.
//
// Anything with compiled code is refused by construction — `py3-none-any` is
// the only tag accepted — because a wheel with a `.so` in it cannot be loaded
// on iOS anyway (extension modules have to be frameworks, signed alongside the
// app). Saying so at install time is much kinder than an ImportError later.

import Foundation

struct Edge0PythonPackage: Identifiable, Hashable {
    var name: String
    var version: String
    /// Shipped inside the app bundle, therefore read-only.
    var bundled: Bool
    var distInfo: String

    var id: String { (bundled ? "bundled:" : "user:") + name }
}

// MARK: - Version ordering

/// Enough of PEP 440 to sort releases: plain dotted numbers only. Anything
/// with a letter in it (rc, b1, dev, post) sorts as "not a candidate" rather
/// than being guessed at — an alpha silently installed because it happened to
/// be newest is a worse failure than not finding one.
enum Edge0Version {
    static func parts(_ text: String) -> [Int]? {
        let pieces = text.split(separator: ".", omittingEmptySubsequences: false)
        guard !pieces.isEmpty else { return nil }
        var numbers: [Int] = []
        for piece in pieces {
            guard let value = Int(piece) else { return nil }
            numbers.append(value)
        }
        return numbers
    }

    static func isRelease(_ text: String) -> Bool { parts(text) != nil }

    static func less(_ lhs: [Int], _ rhs: [Int]) -> Bool {
        for index in 0..<max(lhs.count, rhs.count) {
            let a = index < lhs.count ? lhs[index] : 0
            let b = index < rhs.count ? rhs[index] : 0
            if a != b { return a < b }
        }
        return false
    }

    /// Evaluates a `Requires-Python` specifier set against the interpreter we
    /// actually ship. Unparseable pieces are treated as satisfied: the index
    /// is full of odd specifiers and refusing on one would block packages that
    /// install perfectly well.
    static func satisfies(requiresPython specifier: String, interpreter: [Int]) -> Bool {
        let trimmed = specifier.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return true }
        for clause in trimmed.split(separator: ",") {
            let piece = clause.trimmingCharacters(in: .whitespaces)
            let operators = ["!=", ">=", "<=", "==", "~=", ">", "<"]
            guard let op = operators.first(where: { piece.hasPrefix($0) }) else { continue }
            var bound = String(piece.dropFirst(op.count)).trimmingCharacters(in: .whitespaces)
            let wildcard = bound.hasSuffix(".*")
            if wildcard { bound = String(bound.dropLast(2)) }
            guard let want = parts(bound) else { continue }
            let have = Array(interpreter.prefix(wildcard ? want.count : interpreter.count))
            let equal = !less(have, want) && !less(want, have)
            let ok: Bool
            switch op {
            case ">=": ok = equal || less(want, have)
            case ">": ok = less(want, have)
            case "<=": ok = equal || less(have, want)
            case "<": ok = less(have, want)
            case "==": ok = equal
            case "!=": ok = !equal
            case "~=": ok = equal || less(want, have)
            default: ok = true
            }
            if !ok { return false }
        }
        return true
    }
}

// MARK: - The index

enum Edge0PyPI {
    struct Wheel {
        let package: String
        let version: String
        let filename: String
        let url: URL
    }

    enum Failure: LocalizedError {
        case notFound(String)
        case noPureWheel(String)
        case badIndex(String)

        var errorDescription: String? {
            switch self {
            case .notFound(let name): "PyPI'de \(name) diye bir paket yok."
            case .noPureWheel(let name):
                "\(name) için saf Python (py3-none-any) tekerleği yok — "
                    + "derlenmiş uzantılar iOS'ta kurulamaz."
            case .badIndex(let why): "PyPI dizini okunamadı: \(why)"
            }
        }
    }

    /// PEP 503 name normalisation, which is what the index is keyed by.
    static func normalise(_ name: String) -> String {
        var out = ""
        var previousDash = false
        for character in name.lowercased() {
            if character == "-" || character == "_" || character == "." {
                if !previousDash { out.append("-") }
                previousDash = true
            } else {
                out.append(character)
                previousDash = false
            }
        }
        return out
    }

    /// Every pure-Python wheel the index lists for `name`, newest first.
    static func wheels(for name: String, interpreter: [Int]) async throws -> [Wheel] {
        let normalised = normalise(name)
        guard let url = URL(string: "https://pypi.org/simple/\(normalised)/") else {
            throw Failure.notFound(name)
        }
        var request = URLRequest(url: url)
        request.setValue("application/vnd.pypi.simple.v1+json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 30

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse {
            guard http.statusCode != 404 else { throw Failure.notFound(name) }
            guard http.statusCode == 200 else {
                throw Failure.badIndex("HTTP \(http.statusCode)")
            }
        }
        guard
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let files = root["files"] as? [[String: Any]]
        else { throw Failure.badIndex("beklenmeyen biçim") }

        var found: [Wheel] = []
        for file in files {
            guard
                let filename = file["filename"] as? String,
                filename.hasSuffix("-none-any.whl"),
                let link = file["url"] as? String,
                let url = URL(string: link)
            else { continue }
            // Yanked releases carry either `true` or a reason string.
            if let yanked = file["yanked"] as? Bool, yanked { continue }
            if file["yanked"] is String { continue }
            if let requires = file["requires-python"] as? String,
                !Edge0Version.satisfies(requiresPython: requires, interpreter: interpreter)
            { continue }
            let pieces = filename.split(separator: "-")
            guard pieces.count >= 5 else { continue }
            let version = String(pieces[1])
            guard Edge0Version.isRelease(version) else { continue }
            found.append(
                Wheel(package: String(pieces[0]), version: version, filename: filename, url: url))
        }
        guard !found.isEmpty else { throw Failure.noPureWheel(name) }
        return found.sorted {
            Edge0Version.less(
                Edge0Version.parts($1.version) ?? [], Edge0Version.parts($0.version) ?? [])
        }
    }

    /// The wheel to install: the exact version if one was asked for, the
    /// newest otherwise.
    static func wheel(for name: String, version: String?, interpreter: [Int]) async throws -> Wheel {
        let all = try await wheels(for: name, interpreter: interpreter)
        guard let version else {
            guard let first = all.first else { throw Failure.noPureWheel(name) }
            return first
        }
        guard let match = all.first(where: { $0.version == version }) else {
            throw Failure.noPureWheel("\(name)==\(version)")
        }
        return match
    }
}

// MARK: - The store

@MainActor
@Observable
final class Edge0PythonPackages {
    private(set) var installed: [Edge0PythonPackage] = []
    private(set) var bundled: [Edge0PythonPackage] = []
    private(set) var busy = false
    private(set) var transcript: String = ""

    /// A cap, not a resolver. Dependency graphs on PyPI can be wide, and a
    /// runaway install on a phone over cellular is not something the user can
    /// easily interrupt.
    private let installLimit = 16

    func refresh() {
        bundled = Self.scan(Self.bundledPackages, bundled: true)
        installed = Self.scan(Edge0Python.sitePackages, bundled: false)
    }

    // MARK: Install

    func install(_ request: String) async {
        let trimmed = request.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !busy else { return }
        busy = true
        transcript = ""
        defer { busy = false }

        let (name, version) = Self.split(requirement: trimmed)
        var queue: [(String, String?)] = [(name, version)]
        var seen: Set<String> = []
        var installs = 0
        let interpreter = await Self.interpreterVersion()

        while !queue.isEmpty {
            let (target, wanted) = queue.removeFirst()
            let key = Edge0PyPI.normalise(target)
            guard !seen.contains(key) else { continue }
            seen.insert(key)

            if wanted == nil, isPresent(key) {
                say("• \(target) zaten var, atlandı")
                continue
            }
            guard installs < installLimit else {
                say("• sınır: \(installLimit) paket kuruldu, \(target) atlandı")
                continue
            }

            do {
                let wheel = try await Edge0PyPI.wheel(
                    for: target, version: wanted, interpreter: interpreter)
                say("↓ \(wheel.package) \(wheel.version)")
                let file = try await Self.download(wheel)
                defer { try? FileManager.default.removeItem(at: file) }
                let result = try await Self.unpack(wheel: file, name: wheel.package)
                installs += 1
                say("✓ \(wheel.package) \(wheel.version) — \(result.fileCount) dosya")
                Edge0Log.write("python paketi kuruldu: \(wheel.package) \(wheel.version)")
                for dependency in result.requires where !isPresent(Edge0PyPI.normalise(dependency))
                {
                    queue.append((dependency, nil))
                }
                refresh()
            } catch {
                say("✗ \(target): \(error.localizedDescription)")
                Edge0Log.write("python paketi kurulamadı: \(target) — \(error)")
            }
        }
        refresh()
    }

    // MARK: Remove

    func remove(_ package: Edge0PythonPackage) {
        guard !package.bundled else { return }
        let root = Edge0Python.sitePackages
        let manager = FileManager.default
        let info = root.appendingPathComponent(package.distInfo, isDirectory: true)
        var removed = 0

        if let record = try? String(contentsOf: info.appendingPathComponent("RECORD"), encoding: .utf8) {
            for line in record.split(separator: "\n") {
                guard let path = line.split(separator: ",").first.map(String.init) else { continue }
                // Never let a manifest walk out of site-packages.
                guard !path.hasPrefix("/"), !path.contains("..") else { continue }
                let target = root.appendingPathComponent(path)
                if manager.fileExists(atPath: target.path) {
                    try? manager.removeItem(at: target)
                    removed += 1
                }
            }
        } else if let tops = try? String(
            contentsOf: info.appendingPathComponent("top_level.txt"), encoding: .utf8)
        {
            for line in tops.split(separator: "\n") {
                let top = line.trimmingCharacters(in: .whitespaces)
                guard !top.isEmpty, !top.contains("/"), !top.contains("..") else { continue }
                try? manager.removeItem(at: root.appendingPathComponent(top, isDirectory: true))
                try? manager.removeItem(at: root.appendingPathComponent(top + ".py"))
                removed += 1
            }
        }
        try? manager.removeItem(at: info)
        Self.pruneEmptyDirectories(under: root)
        transcript = "\(package.name) silindi (\(removed) dosya). Etkili olması için uygulamayı yeniden başlatın."
        Edge0Log.write("python paketi silindi: \(package.name) \(package.version)")
        refresh()
    }

    // MARK: Plumbing

    private func say(_ line: String) {
        transcript += (transcript.isEmpty ? "" : "\n") + line
    }

    private func isPresent(_ normalised: String) -> Bool {
        (installed + bundled).contains { Edge0PyPI.normalise($0.name) == normalised }
    }

    private static var bundledPackages: URL {
        Bundle.main.bundleURL.appendingPathComponent("app_packages", isDirectory: true)
    }

    /// `name==version`, or just a name. Other specifiers (`>=`, extras) are
    /// deliberately not honoured: pretending to resolve a range and then
    /// installing the newest thing would be a lie.
    private static func split(requirement: String) -> (String, String?) {
        if let range = requirement.range(of: "==") {
            let name = String(requirement[..<range.lowerBound])
                .trimmingCharacters(in: .whitespaces)
            let version = String(requirement[range.upperBound...])
                .trimmingCharacters(in: .whitespaces)
            return (name, version.isEmpty ? nil : version)
        }
        return (requirement, nil)
    }

    private static func download(_ wheel: Edge0PyPI.Wheel) async throws -> URL {
        var request = URLRequest(url: wheel.url)
        request.timeoutInterval = 120
        let (temporary, response) = try await URLSession.shared.download(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw Edge0PyPI.Failure.badIndex("tekerlek indirilemedi, HTTP \(http.statusCode)")
        }
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent(wheel.filename)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporary, to: destination)
        return destination
    }

    struct Unpacked {
        var fileCount: Int
        var requires: [String]
    }

    enum InstallError: LocalizedError {
        case python(String)

        var errorDescription: String? {
            switch self {
            case .python(let text): text
            }
        }
    }

    /// Unzips through Python's own `zipfile`, which is the one unzipper we are
    /// certain matches the format, and reads back what the wheel declares.
    private static func unpack(wheel: URL, name: String) async throws -> Unpacked {
        let script = """
            import json, os, zipfile, importlib
            wheel = \(Edge0Python.literal(wheel.path))
            dest = \(Edge0Python.literal(Edge0Python.sitePackages.path))
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                archive.extractall(dest)
            info = None
            for entry in names:
                head = entry.split("/")[0]
                if head.endswith(".dist-info"):
                    info = head
                    break
            requires = []
            if info:
                meta = os.path.join(dest, info, "METADATA")
                if os.path.exists(meta):
                    with open(meta, encoding="utf-8", errors="replace") as handle:
                        for line in handle:
                            if not line.strip():
                                break
                            if line.startswith("Requires-Dist:"):
                                requires.append(line.split(":", 1)[1].strip())
            importlib.invalidate_caches()
            _edge0_write(json.dumps({"count": len(names), "requires": requires}))
            """
        let output = try await Task.detached { try Edge0Python.run(script) }.value
        guard !output.isEmpty else { throw InstallError.python("Python yanıt vermedi") }
        guard !output.hasPrefix("HATA") else {
            throw InstallError.python(String(output.prefix(400)))
        }
        guard
            let data = output.data(using: .utf8),
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let count = root["count"] as? Int
        else { throw InstallError.python("açma sonucu okunamadı") }
        let declared = (root["requires"] as? [String]) ?? []
        return Unpacked(fileCount: count, requires: declared.compactMap(dependencyName))
    }

    /// The runtime name out of a `Requires-Dist` line, or nil when the line is
    /// conditional on something we are not (an extra, another Python).
    static func dependencyName(from line: String) -> String? {
        let pieces = line.split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)
        if pieces.count == 2 {
            let marker = pieces[1]
            if marker.contains("extra") { return nil }
            if marker.contains("python_version") {
                guard marker.contains("sys_platform") == false else { return nil }
                if !evaluatePythonVersion(marker: String(marker)) { return nil }
            } else if marker.contains("sys_platform") || marker.contains("platform_system") {
                // ios is neither win32 nor darwin-with-a-desktop; skip rather
                // than guess.
                return nil
            }
        }
        var name = ""
        for character in pieces[0].trimmingCharacters(in: .whitespaces) {
            if character.isLetter || character.isNumber || character == "-" || character == "_"
                || character == "."
            {
                name.append(character)
            } else {
                break
            }
        }
        return name.isEmpty ? nil : name
    }

    /// `python_version < "3.8"` and friends, against 3.14.
    private static func evaluatePythonVersion(marker: String) -> Bool {
        guard let range = marker.range(of: "python_version") else { return true }
        let rest = marker[range.upperBound...].trimmingCharacters(in: .whitespaces)
        let operators = ["!=", ">=", "<=", "==", ">", "<"]
        guard let op = operators.first(where: { rest.hasPrefix($0) }) else { return true }
        let tail = rest.dropFirst(op.count).trimmingCharacters(in: .whitespaces)
        let quoted = tail.drop(while: { $0 == "\"" || $0 == "'" })
        let bound = String(quoted.prefix(while: { $0.isNumber || $0 == "." }))
        return Edge0Version.satisfies(requiresPython: op + bound, interpreter: [3, 14])
    }

    /// Asks the interpreter its own version so the index filter matches what
    /// will actually import the package. Falls back to the version the build
    /// ships when the interpreter is not up.
    private static func interpreterVersion() async -> [Int] {
        let text = await Task.detached { () -> String in
            (try? Edge0Python.run("_edge0_write('%d.%d' % sys.version_info[:2])")) ?? ""
        }.value
        return Edge0Version.parts(text.trimmingCharacters(in: .whitespacesAndNewlines)) ?? [3, 14]
    }

    /// Everything with a `.dist-info` beside it, which is what an installed
    /// package looks like whether pip put it there or we did.
    private static func scan(_ directory: URL, bundled: Bool) -> [Edge0PythonPackage] {
        let entries =
            (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        var packages: [Edge0PythonPackage] = []
        for entry in entries where entry.hasSuffix(".dist-info") {
            let stem = String(entry.dropLast(".dist-info".count))
            let pieces = stem.split(separator: "-")
            guard pieces.count >= 2 else { continue }
            packages.append(
                Edge0PythonPackage(
                    name: pieces.dropLast().joined(separator: "-"),
                    version: String(pieces[pieces.count - 1]),
                    bundled: bundled,
                    distInfo: entry))
        }
        return packages.sorted { $0.name.lowercased() < $1.name.lowercased() }
    }

    private static func pruneEmptyDirectories(under root: URL) {
        let manager = FileManager.default
        guard
            let enumerator = manager.enumerator(
                at: root, includingPropertiesForKeys: [.isDirectoryKey])
        else { return }
        let directories = enumerator.compactMap { $0 as? URL }.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        // Deepest first, so a directory emptied by its children is seen empty.
        for directory in directories.sorted(by: { $0.pathComponents.count > $1.pathComponents.count })
        {
            let contents = (try? manager.contentsOfDirectory(atPath: directory.path)) ?? []
            if contents.isEmpty { try? manager.removeItem(at: directory) }
        }
    }
}
