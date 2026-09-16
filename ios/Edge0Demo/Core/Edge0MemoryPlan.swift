// How many experts each layer may keep in memory — decided by the device, not
// by a number someone picked once.
//
// The setting this replaces was a megabyte figure computed at first launch from
// whatever happened to be free at that moment, then frozen in UserDefaults and
// applied as a hard ceiling for the life of the install. It went wrong in both
// directions inside one afternoon: reinstalling the app on a busy phone came
// back with 2215 MB where the previous install had 2983, which quietly cut the
// cache from 44 slots a layer to 32; raising it to 3323 pushed past what the
// device tolerates and the run spent itself being warned about memory. Neither
// number was wrong when it was chosen. Both were wrong by the time they were
// used, and nothing in the app could tell.
//
// Everything needed to decide this properly is already known at load time: what
// iOS will let this process have, how large one expert of THIS checkpoint
// actually is, and how many layers there are. What a formula cannot know is
// where the line is on a particular device, so the plan also remembers. A run
// that genuinely ran short of memory writes a lower ceiling for its tier; a run
// that completed without one lets the ceiling creep back up. The device gets to
// answer, and the answer is kept.

import Foundation

struct Edge0MemoryPlan {
    /// Experts each layer may cache.
    let slotsPerLayer: Int
    /// What decided it, for the log and for the Settings readout.
    let detail: String
    /// Total memory those slots can occupy across every layer.
    let budgetBytes: Int
}

enum Edge0MemoryPlanner {

    /// Left alone for the KV cache, the activations and MLX's own transient
    /// graph, none of which exist yet when the plan is made.
    ///
    /// Measured rather than guessed, and revised once already: with the
    /// resident weights in and about 5.35 GB left to the process, caches of
    /// 3.30, 2.77 and 2.70 GB all ran the phone short — the last of those with
    /// 2.5 GB nominally held back, which is how this number got bigger. What it
    /// is really compensating for is that `os_proc_available_memory` is
    /// generous: it reported 6.98 GB on a device that started warning once MLX
    /// held 5.1. The remembered ceiling below covers the rest of the distance,
    /// and covers it per device rather than per guess.
    private static let graphReserveBytes = 3_072 * 1024 * 1024

    /// A cache smaller than this is not worth calling one — a decode step alone
    /// routes to K experts per layer.
    private static let floorSlots = 8

    /// Nothing is gained past this; the routing stops repeating first.
    private static let ceilingSlots = 96

    // MARK: Plan

    static func make(
        tier: Edge0Tier,
        automatic: Bool,
        manualBudgetBytes: Int,
        perExpertBytes: Int,
        layerCount: Int,
        alsoReserving reservedBytes: Int = 0
    ) -> Edge0MemoryPlan {
        let perLayerCost = max(1, perExpertBytes)
        let slotsFor: (Int) -> Int = { budget in
            budget / max(1, layerCount * perLayerCost)
        }

        let available = Int(ModelManager.availableProcessMemoryBytes)
        let affordable = max(0, available - graphReserveBytes - max(0, reservedBytes))

        guard automatic else {
            // Manual still gets the memory clamp: the point of manual is to
            // choose, not to be allowed to ask for memory that is not there.
            let effective = min(manualBudgetBytes, affordable)
            let slots = clamp(slotsFor(effective))
            return Edge0MemoryPlan(
                slotsPerLayer: slots,
                detail: "elle \(manualBudgetBytes / 1_048_576) MB"
                    + (manualBudgetBytes > affordable
                        ? ", bellek \(affordable / 1_048_576) MB'a kırptı" : ""),
                budgetBytes: slots * layerCount * perLayerCost)
        }

        let fromMemory = clamp(slotsFor(affordable))
        let remembered = rememberedSlots(for: tier)
        let slots: Int
        let detail: String
        if remembered > 0 {
            // Creep back up rather than jumping: whatever made the device
            // short of memory last time may still be running.
            let allowed = remembered + max(1, remembered / 8)
            slots = clamp(min(fromMemory, allowed))
            detail =
                "otomatik · bellek \(fromMemory) slot verirdi,"
                + " cihaz son seferinde \(remembered) slotta durdu"
        } else {
            slots = fromMemory
            detail = "otomatik · \(affordable / 1_048_576) MB kullanılabilir"
        }
        return Edge0MemoryPlan(
            slotsPerLayer: slots, detail: detail,
            budgetBytes: slots * layerCount * perLayerCost)
    }

    private static func clamp(_ slots: Int) -> Int {
        min(ceilingSlots, max(floorSlots, slots))
    }

    // MARK: What the device said

    /// The tier ran short and was cut back to `slots`, which is the size it
    /// then went on to survive at. That is the number worth keeping — the one
    /// that failed is known to fail, and anything between the two is a guess.
    ///
    /// Together with the eighth-at-a-time climb in `make`, this is additive
    /// increase and multiplicative decrease: back off hard from a real limit,
    /// return to it slowly. Written through immediately, because the failure
    /// this protects against is the process being killed outright, and a lesson
    /// still sitting in memory when that happens is a lesson not learned.
    static func recordPressure(tier: Edge0Tier, survivingAt slots: Int) {
        let remembered = rememberedSlots(for: tier)
        guard remembered == 0 || slots < remembered else { return }
        store(max(floorSlots, slots), for: tier)
    }

    /// The tier loaded and answered at `slots` without running short.
    static func recordClean(tier: Edge0Tier, at slots: Int) {
        guard slots > rememberedSlots(for: tier) else { return }
        store(slots, for: tier)
    }

    static func rememberedSlots(for tier: Edge0Tier) -> Int {
        UserDefaults.standard.integer(forKey: key(tier))
    }

    static func forget(tier: Edge0Tier) {
        UserDefaults.standard.removeObject(forKey: key(tier))
    }

    private static func store(_ slots: Int, for tier: Edge0Tier) {
        let defaults = UserDefaults.standard
        defaults.set(slots, forKey: key(tier))
        defaults.synchronize()
    }

    private static func key(_ tier: Edge0Tier) -> String {
        "edge0.autoSlots.\(tier.rawValue)"
    }
}
