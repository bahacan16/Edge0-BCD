// Where a decode step's time actually goes.
//
// The streaming tier has three plausible costs and they are easy to confuse
// with each other: reading experts from storage, the prerouter's own head
// batch, and everything else (attention, the shared expert, sampling). Tokens
// per second alone cannot tell them apart, which means every change to this
// path gets judged on one number that moves for several reasons at once — and a
// change that made one part twice as fast while making another three times
// slower reads exactly like "it got slower".
//
// So each part is timed separately, with counters cheap enough to leave on:
// two lock acquisitions per layer per token against reads measured in
// milliseconds.

import Foundation

enum Edge0Meter {
    private static let lock = NSLock()

    private static var _expertSeconds = 0.0
    private static var _stageSeconds = 0.0
    private static var _prefetchSeconds = 0.0
    private static var _prefetchedRanges = 0
    private static var _reliefs = 0

    /// Time the generation thread spent getting expert weights ready — the
    /// page-ins it could not avoid plus the copies out of the mapping.
    static func addExpertTime(_ seconds: Double) {
        lock.withLock { _expertSeconds += seconds }
    }

    /// Time inside the prerouter's step boundary: the head batch, its two
    /// evals, and reading the predicted expert ids back to the CPU.
    static func addStageTime(_ seconds: Double) {
        lock.withLock { _stageSeconds += seconds }
    }

    /// Background page-ins issued from a prediction. Counted because a
    /// prefetch that reads a lot and saves nothing looks identical to no
    /// prefetch at all from the outside, except slower.
    static func addPrefetch(seconds: Double, ranges: Int) {
        lock.withLock {
            _prefetchSeconds += seconds
            _prefetchedRanges += ranges
        }
    }

    /// A memory warning arrived and the expert caches were cut back. Counted
    /// because it changes what every later number in the run means: a hit rate
    /// measured across one of these is not the cache's hit rate, it is the
    /// average of before and after.
    static func countRelief() {
        lock.withLock { _reliefs += 1 }
    }

    static func reset() {
        lock.withLock {
            _expertSeconds = 0
            _stageSeconds = 0
            _prefetchSeconds = 0
            _prefetchedRanges = 0
            _reliefs = 0
        }
    }

    static var snapshot:
        (expert: Double, stage: Double, prefetch: Double, ranges: Int, reliefs: Int)
    {
        lock.withLock {
            (_expertSeconds, _stageSeconds, _prefetchSeconds, _prefetchedRanges, _reliefs)
        }
    }

    /// One line saying where the time went and how well the cache did.
    ///
    /// Both halves belong together: expert-read time without the hit rate says
    /// how long the reads took but not whether they should have happened at
    /// all, and a prefetch that reads a lot while the hit rate stays flat is
    /// the signature of speculative work that is not paying for itself.
    static func report(prerouter: Bool) -> String {
        let meter = snapshot
        var line = String(
            format:
                "zaman: expert okuma %.2f sn · prerouter başları %.2f sn"
                + " · önden okuma %.2f sn (%d aralık) · prerouter %@",
            meter.expert, meter.stage, meter.prefetch, meter.ranges,
            prerouter ? "açık" : "kapalı")

        let statistics = Edge0ExpertCaches.statistics
        let total = statistics.hits + statistics.misses
        if total > 0 {
            line += String(
                format: " · isabet %%%.0f (%d/%d)",
                Double(statistics.hits) / Double(total) * 100, statistics.hits, total)
        }
        if meter.reliefs > 0 {
            line += " · bellek uyarısı \(meter.reliefs)× (isabet bundan düşük)"
        }
        line += " · MLX tepe \(ModelManager.formatBytes(ModelManager.mlxPeakMemoryBytes))"
        return line
    }

    /// Runs `body`, adding its wall time to `counter`.
    @inline(__always)
    static func measure<T>(_ counter: (Double) -> Void, _ body: () throws -> T) rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        defer { counter(CFAbsoluteTimeGetCurrent() - start) }
        return try body()
    }
}
