// Zero-copy byte-range access into safetensors shards — a Swift port of
// edge0/src/edge0/streaming/mmap.py (Edge0-AI/edge0, Apache-2.0).
//
// Why this exists: MLX materializes a whole safetensors file when it loads it.
// The 35B tier's checkpoint is ~23 GB, so it can never be resident on a phone.
// A single expert, though, is one contiguous byte slice of a stacked
// `[num_experts, ...]` tensor, so the streaming MoE layer maps the shard once
// and copies only the slices the router actually picked.
//
// The mapping is read-only and file-backed: those pages are clean, so iOS can
// evict them under pressure and they do not count against the app's dirty
// memory footprint the way a normal allocation would.

import Darwin
import Foundation
import MLX

struct SafetensorsEntry {
    let name: String
    /// Absolute offset of the tensor's first byte within the file.
    let offset: Int
    let byteCount: Int
    let dtype: String
    let shape: [Int]

    var elementCount: Int { shape.reduce(1, *) }
}

enum SafetensorsError: LocalizedError {
    case cannotOpen(String)
    case cannotMap(String)
    case malformedHeader(String)
    case unknownTensor(String)
    case unsupportedDType(tensor: String, dtype: String)

    var errorDescription: String? {
        switch self {
        case .cannotOpen(let path): "Dosya açılamadı: \(path)"
        case .cannotMap(let path): "Dosya belleğe eşlenemedi: \(path)"
        case .malformedHeader(let path): "Bozuk safetensors başlığı: \(path)"
        case .unknownTensor(let name): "Tensör bulunamadı: \(name)"
        case .unsupportedDType(let tensor, let dtype):
            "Desteklenmeyen tensör türü \(dtype): \(tensor)"
        }
    }
}

/// One safetensors shard, mapped read-only.
final class SafetensorsMmap {
    let url: URL
    private(set) var entries: [String: SafetensorsEntry] = [:]

    private let descriptor: Int32
    private let base: UnsafeRawPointer
    private let length: Int

    init(url: URL) throws {
        self.url = url

        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw SafetensorsError.cannotOpen(url.path) }
        self.descriptor = descriptor

        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 16 else {
            close(descriptor)
            throw SafetensorsError.cannotOpen(url.path)
        }
        length = Int(status.st_size)

        guard
            let mapped = mmap(nil, length, PROT_READ, MAP_FILE | MAP_SHARED, descriptor, 0),
            mapped != MAP_FAILED
        else {
            close(descriptor)
            throw SafetensorsError.cannotMap(url.path)
        }
        base = UnsafeRawPointer(mapped)

        // Random access across a multi-GB file: tell the kernel not to read
        // ahead as if this were a sequential scan.
        madvise(UnsafeMutableRawPointer(mutating: base), length, MADV_RANDOM)

        do {
            try parseHeader()
        } catch {
            munmap(UnsafeMutableRawPointer(mutating: base), length)
            close(descriptor)
            throw error
        }
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), length)
        close(descriptor)
    }

    private func parseHeader() throws {
        let headerLength = Int(base.loadUnaligned(fromByteOffset: 0, as: UInt64.self))
        guard headerLength > 0, 8 + headerLength <= length else {
            throw SafetensorsError.malformedHeader(url.path)
        }
        let headerData = Data(
            bytes: base.advanced(by: 8), count: headerLength)
        guard
            let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { throw SafetensorsError.malformedHeader(url.path) }

        let payloadBase = 8 + headerLength
        for (name, value) in json {
            guard name != "__metadata__", let meta = value as? [String: Any],
                let offsets = meta["data_offsets"] as? [Int], offsets.count == 2,
                let dtype = meta["dtype"] as? String,
                let shape = meta["shape"] as? [Int]
            else { continue }
            entries[name] = SafetensorsEntry(
                name: name,
                offset: payloadBase + offsets[0],
                byteCount: offsets[1] - offsets[0],
                dtype: dtype,
                shape: shape
            )
        }
    }

    /// Asks the kernel to page in a byte range in the background. Demand
    /// faulting the same range one page at a time serializes on the VM map
    /// lock, which is what makes a cold expert load feel like a stall.
    func prefetch(offset: Int, byteCount: Int) {
        guard byteCount > 0, offset >= 0, offset + byteCount <= length else { return }
        let page = Int(getpagesize())
        let start = (offset / page) * page
        let end = min(length, ((offset + byteCount + page - 1) / page) * page)
        guard end > start else { return }
        madvise(
            UnsafeMutableRawPointer(mutating: base.advanced(by: start)), end - start,
            MADV_WILLNEED)
    }

    /// A raw pointer into the mapping. Only valid while this object is alive.
    func rawPointer(offset: Int, byteCount: Int) throws -> UnsafeRawBufferPointer {
        guard offset >= 0, byteCount >= 0, offset + byteCount <= length else {
            throw SafetensorsError.malformedHeader(url.path)
        }
        return UnsafeRawBufferPointer(start: base.advanced(by: offset), count: byteCount)
    }

    /// Copies one slice of a stacked tensor into an `MLXArray`.
    ///
    /// `rowIndex` selects along the leading (expert) axis; the remaining axes
    /// become the array's shape. Quantized payloads are packed `uint32`, which
    /// is how MLX's quantized kernels want them.
    func slice(
        tensor name: String, rowIndex: Int, as dtype: DType
    ) throws -> MLXArray {
        guard let entry = entries[name] else { throw SafetensorsError.unknownTensor(name) }
        guard let leading = entry.shape.first, leading > 0, rowIndex < leading else {
            throw SafetensorsError.unknownTensor(name)
        }
        let rowBytes = entry.byteCount / leading
        let offset = entry.offset + rowIndex * rowBytes
        let shape = Array(entry.shape.dropFirst())
        let buffer = try rawPointer(offset: offset, byteCount: rowBytes)

        return try array(
            buffer: buffer, shape: shape, byteCount: rowBytes, dtype: dtype, tensor: name)
    }

    /// Copies an entire tensor out of the mapping.
    func whole(tensor name: String, as dtype: DType) throws -> MLXArray {
        guard let entry = entries[name] else { throw SafetensorsError.unknownTensor(name) }
        let buffer = try rawPointer(offset: entry.offset, byteCount: entry.byteCount)
        return try array(
            buffer: buffer, shape: entry.shape, byteCount: entry.byteCount, dtype: dtype,
            tensor: name)
    }

    /// Builds an `MLXArray` over a copy of `buffer`. Shared by the whole-tensor
    /// and single-row readers so the two can never support different dtypes.
    private func array(
        buffer: UnsafeRawBufferPointer, shape: [Int], byteCount: Int, dtype: DType, tensor: String
    ) throws -> MLXArray {
        switch dtype {
        case .uint32: MLXArray(buffer, shape, type: UInt32.self)
        case .int32: MLXArray(buffer, shape, type: Int32.self)
        case .uint16: MLXArray(buffer, shape, type: UInt16.self)
        case .int16: MLXArray(buffer, shape, type: Int16.self)
        case .uint8: MLXArray(buffer, shape, type: UInt8.self)
        case .int8: MLXArray(buffer, shape, type: Int8.self)
        case .float16: MLXArray(buffer, shape, type: Float16.self)
        case .float32: MLXArray(buffer, shape, type: Float.self)
        case .bfloat16:
            // No native Swift bfloat16; reinterpret the raw halves and let MLX
            // view them with the right dtype.
            MLXArray(buffer, [byteCount / 2], type: UInt16.self)
                .view(dtype: .bfloat16).reshaped(shape)
        default:
            throw SafetensorsError.unsupportedDType(tensor: tensor, dtype: "\(dtype)")
        }
    }

    static func dtype(from safetensorsName: String) -> DType? {
        switch safetensorsName {
        case "F32": .float32
        case "F16": .float16
        case "BF16": .bfloat16
        case "U32": .uint32
        case "I32": .int32
        case "U16": .uint16
        case "I16": .int16
        case "U8": .uint8
        case "I8": .int8
        default: nil
        }
    }
}

/// Opens every `*.safetensors` shard in a model directory and indexes which
/// shard holds each tensor.
final class SafetensorsShardSet {
    private var shards: [SafetensorsMmap] = []
    private var index: [String: SafetensorsMmap] = [:]

    init(directory: URL) throws {
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        where file.pathExtension == "safetensors" && !Edge0Adapters.isAdapterFile(file) {
            guard let shard = try? SafetensorsMmap(url: file) else { continue }
            shards.append(shard)
            for name in shard.entries.keys where index[name] == nil {
                index[name] = shard
            }
        }
    }

    var tensorNames: some Collection<String> { index.keys }

    func shard(for tensor: String) -> SafetensorsMmap? { index[tensor] }

    func entry(for tensor: String) -> SafetensorsEntry? {
        index[tensor]?.entries[tensor]
    }
}

/// Tells edge0's adapter tensors apart from the base checkpoint's.
///
/// Both Hugging Face repos ship `lora_edge0_*.safetensors` and
/// `prerouter_edge0_*.safetensors` *inside the model directory* — that is the
/// layout upstream documents. Every loader in this app and in mlx-swift-lm
/// picks up weights by globbing `*.safetensors`, so without this filter the
/// adapters are read as if they were model weights: the LoRA pairs and the
/// prerouter's `layers.<n>.fc1.weight` heads have no counterpart in the module
/// tree, and applying them fails outright.
enum Edge0Adapters {
    /// True for a file that holds adapters rather than base weights.
    static func isAdapterFile(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        return name.hasPrefix("lora_") || name.hasPrefix("prerouter_")
    }

    /// True for a tensor name that came from one of those files.
    static func isAdapterTensor(_ key: String) -> Bool {
        if key.hasSuffix(".lora_A") || key.hasSuffix(".lora_B") { return true }
        if key.contains("prerouter") { return true }
        // Prerouter heads are stored unprefixed as `layers.<n>.fc1.weight`;
        // no base checkpoint key starts with a bare `layers.`.
        return key.hasPrefix("layers.")
    }
}
