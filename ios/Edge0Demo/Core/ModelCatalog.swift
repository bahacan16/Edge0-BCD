import Foundation

/// The two tiers edge0 publishes. Numbers come from the upstream repo's
/// model adapters (`src/edge0/models/edge0_{8b,35b}/__init__.py`) and README.
enum Edge0Tier: String, CaseIterable, Identifiable, Codable, Sendable {
    case edge0_8b = "edge0-8b"
    case edge0_35b = "edge0-35b"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .edge0_8b: "Edge0 8B"
        case .edge0_35b: "Edge0 35B"
        }
    }

    var tagline: String {
        switch self {
        case .edge0_8b: "8B toplam · 1B aktif · 128 expert"
        case .edge0_35b: "35B toplam · 3B aktif · 256 expert"
        }
    }

    var repoId: String {
        switch self {
        case .edge0_8b: "Edge0/Edge0-8B-A1B-preview"
        case .edge0_35b: "Edge0/Edge0-35B-A3B-preview"
        }
    }

    /// `model_type` in the checkpoint's config.json. The 8B tier is Ling 3.0's
    /// hybrid backbone (ported in Edge0BailingHybrid.swift); the 35B tier is
    /// Qwen3.5-MoE, which mlx-swift-lm already implements natively.
    var modelType: String {
        switch self {
        case .edge0_8b: "bailing_hybrid"
        case .edge0_35b: "qwen3_5_moe"
        }
    }

    /// What the checkpoint calls itself in `architectures`, for the tiers whose
    /// config.json carries no `model_type` — edge0's 8B is one.
    var architectureNames: [String] {
        switch self {
        case .edge0_8b: ["BailingMoeV3ForCausalLM", "BailingMoeForCausalLM"]
        case .edge0_35b: ["Qwen3MoeForCausalLM", "Qwen3NextForCausalLM"]
        }
    }

    var loraFileName: String {
        switch self {
        case .edge0_8b: "lora_edge0_8b.safetensors"
        case .edge0_35b: "lora_edge0_35b.safetensors"
        }
    }

    /// edge0's trained prerouter adapter, where this app can use one.
    ///
    /// Both tiers ship one, but the 8B tier's heads are consumed inside Ling's
    /// own MoE block (sigmoid group-limited selection, features read from the
    /// model's caches) rather than replacing the routing from outside, and that
    /// path is not ported. Returning nil is what keeps the 8B tier on its gate
    /// instead of loading heads nothing would consume.
    var prerouterFileName: String? {
        switch self {
        case .edge0_8b: nil
        case .edge0_35b: "prerouter_edge0_35b.safetensors"
        }
    }

    /// First layer that routes from a prediction. The layer below it owns the
    /// first head. Both numbers come from the tier's `PrerouterSpec`.
    var prerouterStartLayer: Int { 7 }

    /// Hidden width of one prerouter head (512 for both shipped tiers).
    var prerouterHiddenSize: Int { 512 }

    /// Approximate on-disk size of the 4-bit checkpoint (README "Requirements").
    var downloadSizeGB: Double {
        switch self {
        case .edge0_8b: 4.2
        case .edge0_35b: 23.0
        }
    }

    /// edge0's own measured peak *active* memory, i.e. with expert streaming on.
    var peakActiveMemoryGB: Double {
        switch self {
        case .edge0_8b: 1.4
        case .edge0_35b: 3.4
        }
    }

    /// Whether the tier can be held fully resident in an iPhone's RAM. The 35B
    /// checkpoint is far larger than any iPhone's memory, so its experts have
    /// to be streamed from storage.
    var requiresExpertStreaming: Bool {
        switch self {
        case .edge0_8b: false
        case .edge0_35b: true
        }
    }

    /// Whether this tier has a streaming expert implementation. Only the
    /// Qwen3.5-MoE backbone is wired for it today; the 8B tier's experts are
    /// small enough to stay resident anyway.
    var supportsExpertStreaming: Bool { requiresExpertStreaming }

    var expertCount: Int {
        switch self {
        case .edge0_8b: 128
        case .edge0_35b: 256
        }
    }

    var expertsPerToken: Int {
        switch self {
        case .edge0_8b: 8
        case .edge0_35b: 4
        }
    }

    var layerCount: Int {
        switch self {
        case .edge0_8b: 24
        case .edge0_35b: 40
        }
    }

    /// Reference decode speed edge0 reports on Apple Silicon — an upper bound
    /// for what to expect on a phone.
    var referenceTokensPerSecond: Double {
        switch self {
        case .edge0_8b: 33
        case .edge0_35b: 13
        }
    }

    var eosTokenIds: Set<Int> {
        switch self {
        case .edge0_8b: [156_895]
        case .edge0_35b: [248_046, 248_044]
        }
    }

    /// Per-tier sampling defaults from each adapter's `GenerationConfig`.
    var defaultSampling: SamplingDefaults {
        switch self {
        case .edge0_8b:
            SamplingDefaults(
                temperature: 0.7, topP: 0.95, topK: 64, repetitionPenalty: 1.1,
                maxTokens: 2048)
        case .edge0_35b:
            SamplingDefaults(
                temperature: 0.7, topP: 0.95, topK: 64, repetitionPenalty: 1.0,
                maxTokens: 2048)
        }
    }

    var accentPair: (start: String, end: String) {
        switch self {
        case .edge0_8b: ("accentCyan", "accentBlue")
        case .edge0_35b: ("accentViolet", "accentPink")
        }
    }
}

struct SamplingDefaults: Sendable {
    var temperature: Float
    var topP: Float
    var topK: Int
    var repetitionPenalty: Float
    var maxTokens: Int
}
