import Foundation
import MLXLMCommon

/// User-facing generation + runtime settings, persisted in `UserDefaults`.
///
/// `@Observable` doesn't compose with `@AppStorage`, so each property writes
/// through on `didSet` and the initializer reads the stored value back.
@Observable
@MainActor
final class AppSettings {
    static let shared = AppSettings()

    // MARK: Model

    var selectedTier: Edge0Tier {
        didSet { store(selectedTier.rawValue, "selectedTier") }
    }

    /// Apply edge0's trained Recover-LoRA adapters. Off = the raw 4-bit base,
    /// which is measurably worse — exposed mostly for A/B curiosity.
    var useLoRA: Bool {
        didSet { store(useLoRA, "useLoRA") }
    }

    /// Stream expert weights from storage instead of holding them in RAM.
    /// Mandatory for the 35B tier; optional (and slower) for 8B.
    var expertStreaming: Bool {
        didSet { store(expertStreaming, "expertStreaming") }
    }

    /// Total RAM the streamed experts may cache, across every MoE layer.
    ///
    /// A decode step touches only K experts per layer, but prompt processing
    /// walks many more, so a cache is what keeps prefill from re-reading the
    /// same weights over and over. The loader turns this into a per-layer slot
    /// count from the checkpoint's real expert size.
    var expertCacheBudgetMB: Int {
        didSet { store(expertCacheBudgetMB, "expertCacheBudgetMB") }
    }

    /// Load the last used tier at launch, if it is already on disk. Only ever
    /// touches a downloaded checkpoint, so it never starts a download by
    /// itself.
    var autoLoadLastModel: Bool {
        didSet { store(autoLoadLastModel, "autoLoadLastModel") }
    }

    // MARK: Sampling

    var temperature: Double {
        didSet { store(temperature, "temperature") }
    }
    var topP: Double {
        didSet { store(topP, "topP") }
    }
    var topK: Int {
        didSet { store(topK, "topK") }
    }
    var repetitionPenalty: Double {
        didSet { store(repetitionPenalty, "repetitionPenalty") }
    }
    var maxTokens: Int {
        didSet { store(maxTokens, "maxTokens") }
    }

    // MARK: Conversation

    var systemPrompt: String {
        didSet { store(systemPrompt, "systemPrompt") }
    }
    /// Qwen/Ling chat templates accept a thinking switch; passed through as
    /// `enable_thinking` in the template context.
    var thinkingMode: Bool {
        didSet { store(thinkingMode, "thinkingMode") }
    }
    var showMetrics: Bool {
        didSet { store(showMetrics, "showMetrics") }
    }
    var hapticsEnabled: Bool {
        didSet { store(hapticsEnabled, "hapticsEnabled") }
    }

    // MARK: Runtime

    /// MLX's buffer cache ceiling in MB. Low values trade a little speed for a
    /// much smaller resident footprint — worth it on a phone.
    var gpuCacheLimitMB: Int {
        didSet { store(gpuCacheLimitMB, "gpuCacheLimitMB") }
    }

    private init() {
        let d = UserDefaults.standard
        let tierRaw = d.string(forKey: Self.key("selectedTier")) ?? Edge0Tier.edge0_8b.rawValue
        let tier = Edge0Tier(rawValue: tierRaw) ?? .edge0_8b
        let defaults = tier.defaultSampling

        selectedTier = tier
        useLoRA = d.object(forKey: Self.key("useLoRA")) as? Bool ?? true
        expertStreaming =
            d.object(forKey: Self.key("expertStreaming")) as? Bool ?? tier.requiresExpertStreaming
        expertCacheBudgetMB =
            d.object(forKey: Self.key("expertCacheBudgetMB")) as? Int
            ?? Self.defaultExpertCacheBudgetMB

        autoLoadLastModel = d.object(forKey: Self.key("autoLoadLastModel")) as? Bool ?? true

        temperature = d.object(forKey: Self.key("temperature")) as? Double
            ?? Double(defaults.temperature)
        topP = d.object(forKey: Self.key("topP")) as? Double ?? Double(defaults.topP)
        topK = d.object(forKey: Self.key("topK")) as? Int ?? defaults.topK
        repetitionPenalty = d.object(forKey: Self.key("repetitionPenalty")) as? Double
            ?? Double(defaults.repetitionPenalty)
        maxTokens = d.object(forKey: Self.key("maxTokens")) as? Int ?? defaults.maxTokens

        systemPrompt = d.string(forKey: Self.key("systemPrompt")) ?? Self.defaultSystemPrompt
        thinkingMode = d.object(forKey: Self.key("thinkingMode")) as? Bool ?? false
        showMetrics = d.object(forKey: Self.key("showMetrics")) as? Bool ?? true
        hapticsEnabled = d.object(forKey: Self.key("hapticsEnabled")) as? Bool ?? true
        gpuCacheLimitMB = d.object(forKey: Self.key("gpuCacheLimitMB")) as? Int ?? 64
    }

    static let defaultSystemPrompt = "Yardımcı, kısa ve net yanıt veren bir asistansın."

    /// A share of what iOS will actually let this process allocate, rather
    /// than of the device's RAM — an app gets only a fraction of the latter,
    /// so sizing against it is how a cache ends up getting the app killed.
    static var defaultExpertCacheBudgetMB: Int {
        let available = Int(ModelManager.availableProcessMemoryBytes) / (1024 * 1024)
        guard available > 0 else { return 256 }
        return max(128, min(1024, available / 8))
    }

    /// Resets sampling to the selected tier's shipped defaults.
    func resetSamplingToTierDefaults() {
        let defaults = selectedTier.defaultSampling
        temperature = Double(defaults.temperature)
        topP = Double(defaults.topP)
        topK = defaults.topK
        repetitionPenalty = Double(defaults.repetitionPenalty)
        maxTokens = defaults.maxTokens
    }

    var generateParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            temperature: Float(temperature),
            topP: Float(topP),
            topK: topK,
            // A penalty of exactly 1.0 is a no-op; leaving it nil skips the
            // repetition pass entirely.
            repetitionPenalty: repetitionPenalty > 1.0 ? Float(repetitionPenalty) : nil
        )
    }

    private static func key(_ name: String) -> String { "edge0.\(name)" }

    private func store(_ value: Any, _ name: String) {
        UserDefaults.standard.set(value, forKey: Self.key(name))
    }
}
