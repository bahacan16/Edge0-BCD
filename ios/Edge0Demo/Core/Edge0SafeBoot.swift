// A one-flag guard against booting into the same crash forever.
//
// The app loads the last used model at launch, which is a convenience right up
// until loading is what kills it. Then the app dies before it can draw the
// switch that would turn the behaviour off, and every launch repeats it — the
// user's only recourse is deleting the app, and with it the 23 GB checkpoint.
//
// So: a flag is set before the load starts and cleared once it survives. If a
// launch finds the flag already set, the previous one did not make it, and this
// one leaves the model alone.

import Foundation

enum Edge0SafeBoot {
    private static let key = "edge0.autoLoadInFlight"

    static var lastAutoLoadCrashed: Bool {
        UserDefaults.standard.bool(forKey: key)
    }

    static func markAutoLoadStarted() {
        UserDefaults.standard.set(true, forKey: key)
        // Written through now: the point of the flag is to outlive a process
        // that is about to be killed without warning.
        UserDefaults.standard.synchronize()
    }

    static func clear() {
        UserDefaults.standard.removeObject(forKey: key)
        UserDefaults.standard.synchronize()
    }
}
