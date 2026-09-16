// Streaming MoE experts — the mechanism that lets a 23 GB checkpoint run on a
// phone. Port of the idea behind edge0/src/edge0/streaming (Edge0-AI/edge0,
// Apache-2.0), rebuilt on MLX Swift's own gather-quantized matmul.
//
// A resident MoE layer keeps every expert's weights in memory: for the 35B tier
// that is ~384 MB per layer, ~15 GB across 40 layers. But one token only routes
// to K experts (4 here), so the working set per step is ~6 MB per layer. This
// module keeps the stacked `[num_experts, out, in]` tensors mapped on disk,
// copies just the selected experts into a small slot buffer, and runs the same
// `gatherQuantizedMM` the resident path would — bit-identical math, a fraction
// of the memory.
//
// Recently used experts are kept in a small per-layer cache, because routing is
// strongly correlated between consecutive tokens.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Quantization layout of one stacked expert tensor set.
struct ExpertQuantization {
    var groupSize: Int
    var bits: Int
}

/// Names of the three stacked projections for one MoE layer, as they appear in
/// the checkpoint.
struct ExpertTensorNames {
    let gateWeight: String
    let gateScales: String
    let gateBiases: String?
    let upWeight: String
    let upScales: String
    let upBiases: String?
    let downWeight: String
    let downScales: String
    let downBiases: String?

    /// Resolves `<modulePath>.{gate,up,down}_proj.{weight,scales,biases}` in the
    /// shard set, tolerating a checkpoint prefix that differs from the module
    /// path (edge0's 35B keys start with `language_model.`).
    init?(modulePath: String, shards: SafetensorsShardSet) {
        func resolve(_ projection: String, _ part: String) -> String? {
            let suffix = "\(projection).\(part)"
            let exact = "\(modulePath).\(suffix)"
            if shards.entry(for: exact) != nil { return exact }
            // Fall back to a suffix match that still pins the layer index.
            guard let key = LayerScopedName(path: modulePath) else { return nil }
            let tail = "layers.\(key.layerIndex).\(key.relativePath).\(suffix)"
            return shards.tensorNames.first { $0.hasSuffix(tail) }
        }

        guard let gateWeight = resolve("gate_proj", "weight"),
            let gateScales = resolve("gate_proj", "scales"),
            let upWeight = resolve("up_proj", "weight"),
            let upScales = resolve("up_proj", "scales"),
            let downWeight = resolve("down_proj", "weight"),
            let downScales = resolve("down_proj", "scales")
        else { return nil }

        self.gateWeight = gateWeight
        self.gateScales = gateScales
        self.gateBiases = resolve("gate_proj", "biases")
        self.upWeight = upWeight
        self.upScales = upScales
        self.upBiases = resolve("up_proj", "biases")
        self.downWeight = downWeight
        self.downScales = downScales
        self.downBiases = resolve("down_proj", "biases")
    }
}

extension ExpertTensorNames {
    /// Bytes one expert occupies across all three projections. Used to turn
    /// the user's cache budget into a slot count — the number that actually
    /// matters is megabytes, and a 35B expert is not the same size as an 8B
    /// one.
    func bytesPerExpert(shards: SafetensorsShardSet) -> Int {
        let tensors = [
            gateWeight, gateScales, gateBiases, upWeight, upScales, upBiases,
            downWeight, downScales, downBiases,
        ].compactMap { $0 }

        var total = 0
        for tensor in tensors {
            guard let shard = shards.shard(for: tensor), let entry = shard.entries[tensor],
                let experts = entry.shape.first, experts > 0
            else { continue }
            total += entry.byteCount / experts
        }
        return max(total, 1)
    }
}

/// One expert's slice of all three projections, materialized in memory.
private struct ExpertSlice {
    let gateWeight: MLXArray
    let gateScales: MLXArray
    let gateBiases: MLXArray?
    let upWeight: MLXArray
    let upScales: MLXArray
    let upBiases: MLXArray?
    let downWeight: MLXArray
    let downScales: MLXArray
    let downBiases: MLXArray?
}

/// One expert's bytes in one projection: a contiguous run of a mapped shard.
struct ExpertByteRange {
    let shard: SafetensorsMmap
    let offset: Int
    let byteCount: Int
}

/// Forces mapped expert bytes resident, as many at a time as the ranges allow.
///
/// This is where the streaming tier's decode speed is won or lost. Each expert
/// is about 1.7 MB of contiguous file, which the device can serve quickly — but
/// faulting them in one after another turns forty layers of four experts into a
/// chain of hundreds of round trips per token, and the measured result is a
/// fraction of what the storage can do. Only POSIX work happens here; MLX is
/// never touched, so this is safe to run off the generation thread.
enum Edge0ExpertPaging {
    static func fault(_ ranges: [ExpertByteRange]) {
        guard ranges.count > 1 else {
            for range in ranges {
                range.shard.fault(offset: range.offset, byteCount: range.byteCount)
            }
            return
        }
        DispatchQueue.concurrentPerform(iterations: ranges.count) { index in
            let range = ranges[index]
            range.shard.fault(offset: range.offset, byteCount: range.byteCount)
        }
    }
}

/// Reads expert slices out of mapped shards, keeping the most recent ones.
final class ExpertSlotPool {
    private let shards: SafetensorsShardSet
    private let names: ExpertTensorNames
    private let capacity: Int

    private var cache: [Int32: ExpertSlice] = [:]
    private var recency: [Int32] = []
    /// Generation runs off the main thread, but a memory warning arrives on it,
    /// so the cache is touched from two threads and needs a lock.
    private let lock = NSLock()

    private var _hits = 0
    private var _misses = 0

    var hits: Int { lock.withLock { _hits } }
    var misses: Int { lock.withLock { _misses } }

    /// Zeroes the counters without touching the cache itself, so one answer's
    /// hit rate can be read on its own rather than blended with every answer
    /// before it.
    func resetStatistics() {
        lock.withLock {
            _hits = 0
            _misses = 0
        }
    }

    /// Drops every cached expert. The next step pages them back in from the
    /// mapping, which is slow but always correct.
    func purge() {
        lock.withLock {
            cache.removeAll()
            recency.removeAll()
        }
    }

    init(shards: SafetensorsShardSet, names: ExpertTensorNames, capacity: Int) {
        self.shards = shards
        self.names = names
        self.capacity = max(1, capacity)
    }

    private func read(_ tensor: String, expert: Int32) throws -> MLXArray {
        guard let shard = shards.shard(for: tensor),
            let entry = shard.entries[tensor],
            let dtype = SafetensorsMmap.dtype(from: entry.dtype)
        else { throw SafetensorsError.unknownTensor(tensor) }
        return try shard.slice(tensor: tensor, rowIndex: Int(expert), as: dtype)
    }

    private func readOptional(_ tensor: String?, expert: Int32) -> MLXArray? {
        guard let tensor else { return nil }
        return try? read(tensor, expert: expert)
    }

    /// Whether an expert is already in memory, so callers can skip the work of
    /// paging in one that is.
    func isCached(_ expert: Int32) -> Bool {
        lock.withLock { cache[expert] != nil }
    }

    /// Where each of these experts' bytes live, skipping the ones already in
    /// memory. Cheap enough to call on the generation thread — it is dictionary
    /// lookups and arithmetic, no I/O.
    func coldRanges(for experts: [Int32]) -> [ExpertByteRange] {
        experts.filter { !isCached($0) }.flatMap(byteRanges(of:))
    }

    private func byteRanges(of expert: Int32) -> [ExpertByteRange] {
        [names.gateWeight, names.upWeight, names.downWeight].compactMap { tensor in
            guard let shard = shards.shard(for: tensor), let entry = shard.entries[tensor],
                let leading = entry.shape.first, leading > 0
            else { return nil }
            let rowBytes = entry.byteCount / leading
            return ExpertByteRange(
                shard: shard, offset: entry.offset + Int(expert) * rowBytes,
                byteCount: rowBytes)
        }
    }

    /// Brings every byte these experts need into memory, several at a time.
    func warm(experts: [Int32]) {
        Edge0ExpertPaging.fault(experts.flatMap(byteRanges(of:)))
    }

    fileprivate func slice(for expert: Int32) throws -> ExpertSlice {
        if let cached = lock.withLock({ () -> ExpertSlice? in
            guard let cached = cache[expert] else { return nil }
            _hits += 1
            touch(expert)
            return cached
        }) {
            return cached
        }
        lock.withLock { _misses += 1 }

        let slice = ExpertSlice(
            gateWeight: try read(names.gateWeight, expert: expert),
            gateScales: try read(names.gateScales, expert: expert),
            gateBiases: readOptional(names.gateBiases, expert: expert),
            upWeight: try read(names.upWeight, expert: expert),
            upScales: try read(names.upScales, expert: expert),
            upBiases: readOptional(names.upBiases, expert: expert),
            downWeight: try read(names.downWeight, expert: expert),
            downScales: try read(names.downScales, expert: expert),
            downBiases: readOptional(names.downBiases, expert: expert)
        )
        lock.withLock {
            cache[expert] = slice
            touch(expert)
            evictIfNeeded()
        }
        return slice
    }

    /// Both callers hold `lock`.
    private func touch(_ expert: Int32) {
        recency.removeAll { $0 == expert }
        recency.append(expert)
    }

    private func evictIfNeeded() {
        while recency.count > capacity {
            let victim = recency.removeFirst()
            cache[victim] = nil
        }
    }
}

/// The selected experts, stacked into one slot buffer per projection.
private struct StackedSlots {
    let gateWeight: MLXArray
    let gateScales: MLXArray
    let gateBiases: MLXArray?
    let upWeight: MLXArray
    let upScales: MLXArray
    let upBiases: MLXArray?
    let downWeight: MLXArray
    let downScales: MLXArray
    let downBiases: MLXArray?
}

/// Every streaming layer built so far, weakly held.
///
/// When iOS warns about memory the app has seconds to give some back, and the
/// expert caches are the largest thing it can drop without losing the model.
/// Walking the module tree to find them would mean touching the model from the
/// main thread while it is generating, so they register here instead.
enum Edge0ExpertCaches {
    private final class WeakLayer {
        weak var layer: Edge0StreamingSwitchGLU?
        init(_ layer: Edge0StreamingSwitchGLU) { self.layer = layer }
    }

    private static var layers: [WeakLayer] = []
    private static let lock = NSLock()

    static func register(_ layer: Edge0StreamingSwitchGLU) {
        lock.withLock {
            layers.removeAll { $0.layer == nil }
            layers.append(WeakLayer(layer))
        }
    }

    /// Drops every layer's cached experts.
    static func purge() {
        let current = lock.withLock { layers.compactMap(\.layer) }
        for layer in current { layer.purge() }
    }

    /// Zeroes every layer's hit/miss counters, leaving the caches themselves
    /// alone.
    static func resetStatistics() {
        let current = lock.withLock { layers.compactMap(\.layer) }
        for layer in current { layer.resetCacheStatistics() }
    }

    /// Aggregate hit/miss counts across every streaming layer.
    static var statistics: (hits: Int, misses: Int) {
        let current = lock.withLock { layers.compactMap(\.layer) }
        return current.reduce(into: (hits: 0, misses: 0)) { total, layer in
            let layerStatistics = layer.cacheStatistics
            total.hits += layerStatistics.hits
            total.misses += layerStatistics.misses
        }
    }

    static var layerCount: Int {
        lock.withLock { layers.compactMap(\.layer).count }
    }
}

/// A `SwitchGLU` whose expert weights live on disk.
///
/// The superclass is initialized with 1×1×1 placeholder projections so it never
/// allocates the full expert tensors; every forward pass goes through the
/// override below, which never touches them.
///
/// - Important: this hinges on `E0Qwen35SparseMoeBlock` calling
///   `switchMLP(x, inds)`, which is what the vendored 3.31.4 copy does. Later
///   mlx-swift-lm versions route through `callAndWeightedReduce` instead, and
///   an override that no longer matches the call site is not a compile error —
///   the block would quietly run the placeholder projections. Anyone
///   re-vendoring Vendor/Edge0Qwen35.swift from a newer release has to check
///   which entry point the block calls and override that one.
final class Edge0StreamingSwitchGLU: E0SwitchGLU {
    private let pool: ExpertSlotPool
    private let quantization: ExpertQuantization
    private let activationFunction: (MLXArray) -> MLXArray
    private var lastSlots: (key: [Int32], slots: StackedSlots)?
    /// Above this many distinct experts in one call, the token dimension is
    /// split and processed in pieces. Without it a long prompt can reference
    /// every expert at once, which would stack the entire layer — the exact
    /// allocation this class exists to avoid.
    ///
    /// It tracks the cache size because the stack is a transient copy on top
    /// of the cache: a fixed ceiling would let a device given a small budget
    /// still allocate the same large stack during prefill, which is where the
    /// budget matters most. The floor keeps it comfortably above the experts
    /// a single token routes to, so splitting always terminates.
    private let maxExpertsPerCall: Int

    init(
        shards: SafetensorsShardSet,
        names: ExpertTensorNames,
        quantization: ExpertQuantization,
        hotSlots: Int
    ) {
        self.pool = ExpertSlotPool(
            shards: shards, names: names, capacity: hotSlots)
        self.quantization = quantization
        self.activationFunction = MLXNN.silu
        self.maxExpertsPerCall = max(8, hotSlots)
        super.init(inputDims: 1, hiddenDims: 1, numExperts: 1, bias: false)
        Edge0ExpertCaches.register(self)
    }

    /// Releases the cached experts. Called on a memory warning.
    func purge() {
        lastSlots = nil
        pool.purge()
    }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        // Reading the router's choice costs a GPU→CPU sync, and this runs once
        // per MoE layer per step, so it is read once and passed down rather
        // than fetched again inside `project`.
        let requested = indices.asArray(Int32.self)
        let tokenCount = indices.dim(-2)
        if tokenCount > 1, Set(requested).count > maxExpertsPerCall {
            return splitOverTokens(x, indices)
        }
        return project(x, indices, requested: requested)
    }

    /// Halves the sequence until each piece touches few enough experts.
    private func splitOverTokens(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let tokenAxis = x.ndim - 2
        let tokenCount = x.dim(tokenAxis)
        let middle = tokenCount / 2
        let first = callAsFunction(
            x[.ellipsis, ..<middle, 0...], indices[.ellipsis, ..<middle, 0...])
        let second = callAsFunction(
            x[.ellipsis, middle..., 0...], indices[.ellipsis, middle..., 0...])
        return MLX.concatenated([first, second], axis: tokenAxis)
    }

    private func project(_ x: MLXArray, _ indices: MLXArray, requested: [Int32]) -> MLXArray {
        let unique = Array(Set(requested)).sorted()
        var slotOf: [Int32: Int32] = [:]
        for (slot, expert) in unique.enumerated() { slotOf[expert] = Int32(slot) }

        let slots: StackedSlots
        do {
            slots = try stackedSlots(for: unique)
        } catch {
            // Losing an expert would silently corrupt the answer, so fail loudly
            // rather than routing through zeros.
            fatalError("edge0 streaming experts: \(error)")
        }

        let slotIndices = MLXArray(
            requested.map { slotOf[$0] ?? 0 }, indices.shape)

        var h = MLX.expandedDimensions(x, axes: [-2, -3])

        let gate = gather(
            h, weight: slots.gateWeight, scales: slots.gateScales, biases: slots.gateBiases,
            indices: slotIndices)
        let up = gather(
            h, weight: slots.upWeight, scales: slots.upScales, biases: slots.upBiases,
            indices: slotIndices)

        h = gather(
            activationFunction(gate) * up,
            weight: slots.downWeight, scales: slots.downScales, biases: slots.downBiases,
            indices: slotIndices)

        return MLX.squeezed(h, axis: -2)
    }

    /// Stacking the selected experts is itself a copy, and consecutive tokens
    /// very often route to the same set, so the last stack is reused.
    private func stackedSlots(for experts: [Int32]) throws -> StackedSlots {
        if let cached = lastSlots, cached.key == experts {
            return cached.slots
        }
        return try Edge0Meter.measure(Edge0Meter.addExpertTime) {
            try buildSlots(for: experts)
        }
    }

    /// Everything a miss costs: the page-ins, the copies out of the mapping,
    /// and one materialisation for the layer.
    private func buildSlots(for experts: [Int32]) throws -> StackedSlots {
        // Only worth paging in once the memo has actually missed, and only the
        // experts that are not already cached — the rest are in memory.
        let cold = experts.filter { !pool.isCached($0) }
        if !cold.isEmpty { pool.warm(experts: cold) }
        var slices: [ExpertSlice] = []
        slices.reserveCapacity(experts.count)
        for expert in experts {
            slices.append(try pool.slice(for: expert))
        }
        // One materialisation for the whole layer rather than one per expert.
        // These are leaves copied out of the mapping, but each `eval` is still
        // a trip through MLX's scheduler, and at forty layers times four
        // experts a token that adds up.
        eval(
            slices.flatMap {
                [$0.gateWeight, $0.gateScales, $0.upWeight, $0.upScales,
                 $0.downWeight, $0.downScales]
            })
        let slots = StackedSlots(
            gateWeight: MLX.stacked(slices.map(\.gateWeight)),
            gateScales: MLX.stacked(slices.map(\.gateScales)),
            gateBiases: stackedOptional(slices.map(\.gateBiases)),
            upWeight: MLX.stacked(slices.map(\.upWeight)),
            upScales: MLX.stacked(slices.map(\.upScales)),
            upBiases: stackedOptional(slices.map(\.upBiases)),
            downWeight: MLX.stacked(slices.map(\.downWeight)),
            downScales: MLX.stacked(slices.map(\.downScales)),
            downBiases: stackedOptional(slices.map(\.downBiases))
        )
        lastSlots = (experts, slots)
        return slots
    }

    private func gather(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        indices: MLXArray
    ) -> MLXArray {
        MLX.gatherQuantizedMM(
            x, weight, scales: scales, biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
    }

    private func stackedOptional(_ arrays: [MLXArray?]) -> MLXArray? {
        let present = arrays.compactMap { $0 }
        guard present.count == arrays.count, !present.isEmpty else { return nil }
        return MLX.stacked(present)
    }

    /// The byte ranges these experts would have to read, skipping what is
    /// already cached. The prerouter uses this to page in a whole predicted
    /// step ahead of the forward that needs it.
    func coldRanges(for experts: [Int32]) -> [ExpertByteRange] {
        pool.coldRanges(for: experts)
    }

    var cacheStatistics: (hits: Int, misses: Int) { (pool.hits, pool.misses) }

    func resetCacheStatistics() { pool.resetStatistics() }
}

/// A module path split at its `layers.<n>.` boundary, used to match checkpoint
/// tensor names against module paths that carry a different prefix.
struct LayerScopedName {
    let layerIndex: Int
    let relativePath: String

    init?(path: String) {
        let parts = path.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count >= 3 else { return nil }
        var cursor = parts.count - 3
        var match: (index: Int, layer: Int)?
        while cursor >= 0 {
            if parts[cursor] == "layers", let layer = Int(parts[cursor + 1]) {
                match = (cursor, layer)
                break
            }
            cursor -= 1
        }
        guard let match else { return nil }
        layerIndex = match.layer
        relativePath = parts[(match.index + 2)...].joined(separator: ".")
    }
}
