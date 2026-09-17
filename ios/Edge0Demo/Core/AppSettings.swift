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

    /// Use edge0's trained prerouter: each layer predicts the next layer's
    /// routing one token ahead and supplies it, which is the configuration the
    /// Recover-LoRA was trained in.
    ///
    /// Off by default, and measured rather than assumed: on this device the
    /// port is currently *slower* than plain gate routing — 27.0 s against
    /// 11.9 s on the fixed health-check workload, with a larger expert cache
    /// than the faster run had. The prediction is sound; the read it enables is
    /// not yet a saving, because the layer still does its own read afterwards
    /// instead of consuming what was fetched for it. Until that handoff exists,
    /// this switch is for measuring, not for speed.
    var usePrerouter: Bool {
        didSet { store(usePrerouter, "usePrerouter") }
    }

    /// Let the app size the expert cache from the device and the checkpoint,
    /// and adjust it from what the device actually does.
    ///
    /// On is the sane default and the manual stepper below only applies when
    /// this is off. The number this replaces was chosen once at first launch
    /// from whatever was free at that moment, frozen, and then applied as a
    /// hard ceiling forever — which went wrong in both directions inside one
    /// afternoon without anything being able to notice.
    var automaticMemoryTuning: Bool {
        didSet { store(automaticMemoryTuning, "automaticMemoryTuning") }
    }

    /// Total RAM the streamed experts may cache, across every MoE layer.
    /// Applies only when `automaticMemoryTuning` is off.
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
    ///
    /// Off by default. Loading a 23 GB checkpoint takes the better part of a
    /// minute and claims most of the phone's memory, which is not what opening
    /// an app should do to you before you have said what you want.
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

    /// Bits per value in the KV cache, or 0 for none.
    ///
    /// The cache is what grows with the conversation, and on a resident model
    /// it is the only thing competing with the weights for memory. Storing it
    /// at 8 bits instead of 16 roughly halves that growth; 4 bits quarters it.
    /// Only the full-attention layers are affected — a hybrid model's linear
    /// layers keep a fixed-size state that is neither large nor quantizable.
    var kvCacheBits: Int {
        didSet { store(kvCacheBits, "kvCacheBits") }
    }

    /// How many tokens stay exact before quantization starts. A short
    /// conversation never reaches it and pays nothing.
    var kvCacheStart: Int {
        didSet { store(kvCacheStart, "kvCacheStart") }
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
        automaticMemoryTuning =
            d.object(forKey: Self.key("automaticMemoryTuning")) as? Bool ?? true
        expertCacheBudgetMB =
            d.object(forKey: Self.key("expertCacheBudgetMB")) as? Int
            ?? Self.defaultExpertCacheBudgetMB

        usePrerouter = d.object(forKey: Self.key("usePrerouter")) as? Bool ?? false
        autoLoadLastModel = d.object(forKey: Self.key("autoLoadLastModel")) as? Bool ?? false

        temperature = d.object(forKey: Self.key("temperature")) as? Double
            ?? Double(defaults.temperature)
        topP = d.object(forKey: Self.key("topP")) as? Double ?? Double(defaults.topP)
        topK = d.object(forKey: Self.key("topK")) as? Int ?? defaults.topK
        repetitionPenalty = d.object(forKey: Self.key("repetitionPenalty")) as? Double
            ?? Double(defaults.repetitionPenalty)
        maxTokens = d.object(forKey: Self.key("maxTokens")) as? Int ?? defaults.maxTokens
        // Off by default: quantizing the cache changes what the model
        // attends to, and that is the user's call to make rather than one to
        // inherit from an update.
        kvCacheBits = d.object(forKey: Self.key("kvCacheBits")) as? Int ?? 0
        kvCacheStart = d.object(forKey: Self.key("kvCacheStart")) as? Int ?? 1024

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
    /// Generous by default: this is the number that decides decode speed on
    /// the streaming tier, and the loader clamps it to what is actually free
    /// once the model is resident, so asking for too much is safe.
    static var defaultExpertCacheBudgetMB: Int {
        let available = Int(ModelManager.availableProcessMemoryBytes) / (1024 * 1024)
        guard available > 0 else { return 1024 }
        // Capped below the size that has been measured to trip a memory
        // warning on this device class. 44 slots per layer (about 2.9 GB across
        // forty layers) ran clean; 49 warned mid-run, and a warning costs more
        // than the extra slots ever earned. The loader still clamps this to
        // what is genuinely free, and the caches now halve themselves if the
        // device disagrees anyway.
        return max(1024, min(2816, available / 2))
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
            kvBits: kvCacheBits > 0 ? kvCacheBits : nil,
            kvGroupSize: 64,
            quantizedKVStart: kvCacheStart,
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
