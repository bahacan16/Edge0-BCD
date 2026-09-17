// The embedded CPython runtime.
//
// The interpreter lives in the app bundle as `Python.xcframework`, its standard
// library under `python/lib/python3.x`, and its two hundred extension modules
// as individual frameworks — iOS will not load unsigned machine code, and a
// `.so` in Resources is not somewhere a signer looks, so the build phase turns
// each one into `Frameworks/<module>.framework` and leaves a `.fwork`
// placeholder behind for CPython's iOS loader to follow.
//
// Started from environment variables rather than `PyConfig`. The struct-and-
// wide-string dance that briefcase does in Objective-C carries badly into
// Swift, and `Py_Initialize` reads PYTHONHOME and PYTHONPATH like any other
// build does. If that turns out to be wrong the first run says so precisely,
// because it reports `sys.path` rather than just failing.
//
// Deliberately not started at launch. Bringing up an interpreter is exactly
// the kind of thing that fails by killing the process, and an app that dies on
// launch cannot be used to turn the feature off — a lesson this project has
// already paid for once with automatic model loading.

import Foundation
import Python

enum Edge0Python {
    private static var started = false
    private static let lock = NSLock()

    /// Where packages installed after the fact go. Inside Documents, so they
    /// survive an app update and are visible in Files.
    static var sitePackages: URL {
        let manager = FileManager.default
        let base = manager.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? manager.temporaryDirectory
        return base.appendingPathComponent("Python/site-packages", isDirectory: true)
    }

    /// The runtime laid out by the build phase.
    private static var home: URL {
        Bundle.main.bundleURL.appendingPathComponent("python", isDirectory: true)
    }

    /// The packages shipped with the app.
    private static var bundledPackages: URL {
        Bundle.main.bundleURL.appendingPathComponent("app_packages", isDirectory: true)
    }

    /// The `pythonX.Y` directory, discovered rather than hard-coded: the
    /// support package's version moves and a stale constant would fail as a
    /// mysterious empty `sys.path`.
    private static func standardLibrary() -> URL? {
        let lib = home.appendingPathComponent("lib", isDirectory: true)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: lib.path)) ?? []
        guard let version = names.filter({ $0.hasPrefix("python3.") }).sorted().last else {
            return nil
        }
        return lib.appendingPathComponent(version, isDirectory: true)
    }

    enum StartError: LocalizedError {
        case runtimeMissing(String)
        case initializationFailed

        var errorDescription: String? {
            switch self {
            case .runtimeMissing(let what): "Python çalışma zamanı eksik: \(what)"
            case .initializationFailed: "Python başlatılamadı."
            }
        }
    }

    /// Brings the interpreter up, once.
    static func start() throws {
        try lock.withLock {
            guard !started else { return }
            guard FileManager.default.fileExists(atPath: home.path) else {
                throw StartError.runtimeMissing("python/ paketin içinde yok")
            }
            guard let stdlib = standardLibrary() else {
                throw StartError.runtimeMissing("python/lib/python3.x bulunamadı")
            }
            try? FileManager.default.createDirectory(
                at: sitePackages, withIntermediateDirectories: true)

            let searchPath = [
                stdlib.path,
                stdlib.appendingPathComponent("lib-dynload").path,
                bundledPackages.path,
                sitePackages.path,
            ].joined(separator: ":")

            setenv("PYTHONHOME", home.path, 1)
            setenv("PYTHONPATH", searchPath, 1)
            // Nothing in the bundle is writable, and a failed .pyc write is a
            // confusing way to learn that.
            setenv("PYTHONDONTWRITEBYTECODE", "1", 1)
            setenv("PYTHONUNBUFFERED", "1", 1)

            Edge0Log.write("python başlatılıyor · home \(home.lastPathComponent) · \(searchPath)")
            Py_Initialize()
            guard Py_IsInitialized() != 0 else { throw StartError.initializationFailed }
            started = true
            Edge0Log.write("python başlatıldı")
        }
    }

    /// Runs `code`, with the interpreter started if it is not already.
    ///
    /// Results come back through a file rather than through the C API. For a
    /// first run that is the right trade: marshalling PyObjects from Swift is
    /// a second thing to get wrong at the same time as the runtime itself, and
    /// this answers the only question that matters — does it run — without it.
    @discardableResult
    static func run(_ code: String) throws -> String {
        try start()
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("python-\(UUID().uuidString).txt")
        let preamble = """
            import sys, traceback
            EDGE0_OUT = \(literal(output.path))
            def _edge0_write(text):
                with open(EDGE0_OUT, "w", encoding="utf-8") as handle:
                    handle.write(text)
            try:

            """
        let body = code.split(separator: "\n", omittingEmptySubsequences: false)
            .map { "    \($0)" }
            .joined(separator: "\n")
        let epilogue = """

            except BaseException:
                _edge0_write("HATA\\n" + traceback.format_exc())
            """

        PyRun_SimpleStringFlags(preamble + body + epilogue, nil)
        defer { try? FileManager.default.removeItem(at: output) }
        return (try? String(contentsOf: output, encoding: .utf8)) ?? ""
    }

    /// Python string literal for a path, so a stray quote cannot end it early.
    static func literal(_ text: String) -> String {
        "\"\(text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\""))\""
    }

    // MARK: Diagnostics

    /// Everything the first run on a device needs to answer, in one go: does
    /// the interpreter come up, is its path right, do the extension modules
    /// load out of their frameworks, and can ezdxf actually read and write.
    static func selfTest() -> String {
        do {
            return try run(
                """
                out = []
                out.append("python " + sys.version.split()[0])
                out.append("prefix " + sys.prefix)
                out.append("path:")
                for entry in sys.path:
                    out.append("  " + entry)
                for name in ("zlib", "binascii", "_struct", "math", "pyparsing", "ezdxf", "pypdf"):
                    try:
                        module = __import__(name)
                        out.append("ok   " + name + " " + str(getattr(module, "__version__", "")))
                    except Exception as exc:
                        out.append("HATA " + name + ": " + repr(exc))
                import os, ezdxf
                target = os.path.join(os.path.dirname(EDGE0_OUT), "probe.dxf")
                doc = ezdxf.new("R2010")
                doc.modelspace().add_line((0, 0), (10, 5))
                doc.modelspace().add_circle((3, 3), 2)
                doc.layers.add("DUVAR", color=3)
                doc.saveas(target)
                back = ezdxf.readfile(target)
                kinds = [e.dxftype() for e in back.modelspace()]
                layers = sorted(l.dxf.name for l in back.layers)
                out.append("dxf yaz/oku: " + str(kinds) + " katmanlar " + str(layers))
                from pypdf import PdfReader, PdfWriter
                pdf = os.path.join(os.path.dirname(EDGE0_OUT), "probe.pdf")
                writer = PdfWriter()
                writer.add_blank_page(width=200, height=200)
                with open(pdf, "wb") as handle:
                    writer.write(handle)
                out.append("pdf yaz/oku: " + str(len(PdfReader(pdf).pages)) + " sayfa")
                out.append("numpy yüklendi mi: " + str("numpy" in sys.modules))
                _edge0_write("\\n".join(out))
                """)
        } catch {
            return "HATA \(error.localizedDescription)"
        }
    }
}
