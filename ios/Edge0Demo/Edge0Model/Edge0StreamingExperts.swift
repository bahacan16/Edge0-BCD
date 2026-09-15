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

    /// Hints the kernel to page in every byte range this expert needs.
    func prefetch(expert: Int32) {
        for tensor in [names.gateWeight, names.upWeight, names.downWeight] {
            guard let shard = shards.shard(for: tensor), let entry = shard.entries[tensor],
                let leading = entry.shape.first, leading > 0
            else { continue }
            let rowBytes = entry.byteCount / leading
            shard.prefetch(offset: entry.offset + Int(expert) * rowBytes, byteCount: rowBytes)
        }
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
        // Materialize now so the mapped pages can go cold again.
        eval(
            slice.gateWeight, slice.gateScales, slice.upWeight, slice.upScales,
            slice.downWeight, slice.downScales)

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
final class Edge0StreamingSwitchGLU: E0SwitchGLU {
    private let pool: ExpertSlotPool
    private let quantization: ExpertQuantization
    private let activationFunction: (MLXArray) -> MLXArray
    private var lastSlots: (key: [Int32], slots: StackedSlots)?

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
        super.init(inputDims: 1, hiddenDims: 1, numExperts: 1, bias: false)
        Edge0ExpertCaches.register(self)
    }

    /// Releases the cached experts. Called on a memory warning.
    func purge() {
        lastSlots = nil
        pool.purge()
    }

    /// Above this many distinct experts in one call, the token dimension is
    /// split and processed in pieces. Without it a long prompt can reference
    /// every expert at once, which would stack the entire layer — the exact
    /// allocation this class exists to avoid.
    private static let maxExpertsPerCall = 32

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let tokenCount = indices.dim(-2)
        if tokenCount > 1, distinctExpertCount(indices) > Self.maxExpertsPerCall {
            return splitOverTokens(x, indices)
        }
        return project(x, indices)
    }

    private func distinctExpertCount(_ indices: MLXArray) -> Int {
        Set(indices.asArray(Int32.self)).count
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

    private func project(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let requested = indices.asArray(Int32.self)
        let unique = Array(Set(requested)).sorted()
        var slotOf: [Int32: Int32] = [:]
        for (slot, expert) in unique.enumerated() { slotOf[expert] = Int32(slot) }

        for expert in unique { pool.prefetch(expert: expert) }

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
        var slices: [ExpertSlice] = []
        slices.reserveCapacity(experts.count)
        for expert in experts {
            slices.append(try pool.slice(for: expert))
        }
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

    var cacheStatistics: (hits: Int, misses: Int) { (pool.hits, pool.misses) }
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
