import MLX
import SwiftUI

@main
struct Edge0DemoApp: App {
    init() {
        MLXActiveMemory.provider = { MLX.GPU.activeMemory }
        Edge0Log.start()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}
