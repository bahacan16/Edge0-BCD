// Registers the "bailing_hybrid" architecture (edge0-8b's base, Ling 3.0)
// with mlx-swift-lm's model-type registry, downloads the tier from
// Hugging Face, and applies edge0's LoRA adapter — all resident in RAM
// (no SSD streaming / prerouter; see Edge0BailingHybrid.swift).

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

enum Edge0Model {
    /// `Edge0/Edge0-8B-A1B-preview` on Hugging Face — base checkpoint,
    /// trained LoRA adapters, and prerouter heads (unused here) shipped
    /// together in one repo.
    static let configuration = ModelConfiguration(
        id: "Edge0/Edge0-8B-A1B-preview",
        eosTokenIds: [156895]
    )

    static let loraFileName = "lora_edge0_8b.safetensors"
    static let loraRank = 16
    static let loraAlpha: Float = 32.0

    private static var registered = false

    static func registerIfNeeded() async {
        guard !registered else { return }
        registered = true
        await LLMTypeRegistry.shared.registerModelType("bailing_hybrid") { data in
            let configuration = try JSONDecoder().decode(
                Edge0BailingConfiguration.self, from: data)
            return Edge0BailingModel(configuration)
        }
    }

    /// Downloads (if needed) and loads edge0-8b with its LoRA adapter
    /// applied, reporting Hugging Face download progress via `onProgress`.
    static func load(
        onProgress: @escaping @Sendable (Progress) -> Void
    ) async throws -> ModelContainer {
        await registerIfNeeded()

        let downloader = #hubDownloader()
        let resolved = try await resolve(
            configuration: configuration, from: downloader, useLatest: false,
            progressHandler: onProgress)

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolved.modelDirectory,
            using: #huggingFaceTokenizerLoader())

        let loraURL = resolved.modelDirectory.appendingPathComponent(loraFileName)
        if FileManager.default.fileExists(atPath: loraURL.path) {
            try await container.perform { context in
                guard let model = context.model as? Edge0BailingModel else {
                    print("[edge0-lora] model is not Edge0BailingModel, skipping adapter")
                    return
                }
                _ = try applyEdge0LoRA(
                    model: model, fileURL: loraURL, r: loraRank, alpha: loraAlpha)
            }
        } else {
            print("[edge0-lora] \(loraFileName) not found in \(resolved.modelDirectory.path)")
        }

        return container
    }
}
