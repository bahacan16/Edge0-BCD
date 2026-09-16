// A log file the user can actually get at.
//
// It lives next to the model folders, in Documents, so it shows up in the
// Files app and in Finder alongside everything else the app exposes.
//
// Every write is appended and flushed immediately. That costs speed and buys
// the only thing that matters here: the most common way this app dies is
// jetsam killing it for using too much memory, which arrives as SIGKILL —
// uncatchable, no handler, no crash report from us. What survives such a death
// is whatever already reached the disk, so the last line in the file is the
// evidence. Breadcrumbs beat a crash handler.
//
// The signal and exception handlers below catch the rarer, catchable deaths.

import Darwin
import Foundation
import UIKit

enum Edge0Log {
    static let fileURL: URL = {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("Edge0.log")
    }()

    /// Held open for the process's lifetime: a crash handler cannot safely open
    /// a file, so the descriptor has to exist before it is needed.
    nonisolated(unsafe) private static var descriptor: Int32 = -1
    private static let lock = NSLock()
    /// Beyond this the log is restarted, so it cannot grow without bound.
    private static let sizeLimit = 4 * 1024 * 1024

    // MARK: Lifecycle

    static func start() {
        lock.withLock {
            guard descriptor < 0 else { return }
            rotateIfNeeded()
            descriptor = open(fileURL.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        }
        installHandlers()

        let device = UIDevice.current
        write("=== açılış ===")
        write("uygulama \(Self.version) · iOS \(device.systemVersion)")
        write("bellek: uygulamaya kalan \(ModelManager.formatBytes(Int64(os_proc_available_memory())))")
        write("belgeler: \(fileURL.deletingLastPathComponent().path)")
    }

    /// Version, build and the commit it came from.
    ///
    /// The commit is the part that earns its place. Every build called itself
    /// "1.0 (1)", so a log sent from the device could not say which one wrote
    /// it, and reconstructing that from how the app behaved is how a page-fault
    /// change got blamed on a prerouter for two rounds.
    private static var version: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        let commit = info?["Edge0Commit"] as? String
        return "\(short) (\(build))" + (commit.map { " · \($0)" } ?? "")
    }

    private static func rotateIfNeeded() {
        let size = (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size])
            .flatMap { $0 as? Int } ?? 0
        if size > sizeLimit { try? FileManager.default.removeItem(at: fileURL) }
    }

    static func clear() {
        lock.withLock {
            if descriptor >= 0 { close(descriptor) }
            try? FileManager.default.removeItem(at: fileURL)
            descriptor = open(fileURL.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        }
        write("=== günlük temizlendi ===")
    }

    // MARK: Writing

    static func write(_ message: String) {
        let line = "\(timestamp) \(message)\n"
        lock.withLock {
            guard descriptor >= 0 else { return }
            _ = line.utf8CString.withUnsafeBufferPointer { buffer -> Int in
                guard let base = buffer.baseAddress else { return 0 }
                // utf8CString includes the terminating NUL, which must not be
                // written into a text file.
                return Darwin.write(descriptor, base, buffer.count - 1)
            }
            // Flushed now, not at some convenient later moment that a SIGKILL
            // will never grant.
            fsync(descriptor)
        }
    }

    /// Records a thrown error with the type name, which `localizedDescription`
    /// alone tends to lose.
    static func failure(_ context: String, _ error: Error) {
        write("HATA \(context): \(type(of: error)) — \(error.localizedDescription)")
    }

    static func memory(_ context: String) {
        write(
            "bellek [\(context)] kalan \(ModelManager.formatBytes(Int64(os_proc_available_memory())))"
                + " · MLX aktif \(ModelManager.formatBytes(Int64(MLXActiveMemory.current)))")
    }

    private static var timestamp: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }

    // MARK: Catchable deaths

    private static func installHandlers() {
        NSSetUncaughtExceptionHandler { exception in
            Edge0Log.write(
                "=== İSTİSNA \(exception.name.rawValue): \(exception.reason ?? "-") ===")
            for symbol in exception.callStackSymbols { Edge0Log.write("  \(symbol)") }
        }
        for signal in [SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE, SIGTRAP] {
            Darwin.signal(signal, edge0HandleSignal)
        }
    }

    /// Only for the signal handler, which must not take a lock or allocate.
    nonisolated(unsafe) static var rawDescriptor: Int32 { descriptor }
}

/// A C function pointer, so it captures nothing.
///
/// Deliberately minimal: inside a signal handler only async-signal-safe calls
/// are legal, which rules out almost all of Foundation. `write` and
/// `backtrace_symbols_fd` are on the safe list.
private func edge0HandleSignal(_ number: Int32) {
    let fd = Edge0Log.rawDescriptor
    if fd >= 0 {
        _ = "\n=== SİNYAL ".withCString { write(fd, $0, strlen($0)) }
        var digits = [UInt8]()
        var value = number
        repeat {
            digits.insert(UInt8(48 + value % 10), at: 0)
            value /= 10
        } while value > 0
        digits.append(UInt8(ascii: "\n"))
        digits.withUnsafeBufferPointer { _ = write(fd, $0.baseAddress, $0.count) }

        var frames = [UnsafeMutableRawPointer?](repeating: nil, count: 64)
        let count = backtrace(&frames, 64)
        backtrace_symbols_fd(&frames, count, fd)
        fsync(fd)
    }
    Darwin.signal(number, SIG_DFL)
    raise(number)
}

/// Reading MLX's counter without importing MLX into the logger.
enum MLXActiveMemory {
    nonisolated(unsafe) static var provider: @Sendable () -> Int = { 0 }
    static var current: Int { provider() }
}
