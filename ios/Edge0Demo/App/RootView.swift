import SwiftUI

enum RootTab: Hashable {
    case chat, models, settings
}

struct RootView: View {
    @State private var models = ModelManager()
    @State private var settings = AppSettings.shared
    @State private var tab: RootTab = .chat

    var body: some View {
        TabView(selection: $tab) {
            ChatView(selectedTab: $tab)
                .tabItem { Label("Sohbet", systemImage: "bubble.left.and.text.bubble.right") }
                .tag(RootTab.chat)

            ModelsView()
                .tabItem { Label("Modeller", systemImage: "square.stack.3d.down.right") }
                .tag(RootTab.models)
                .badge(models.phase.isBusy ? "•" : nil)

            SettingsView()
                .tabItem { Label("Ayarlar", systemImage: "slider.horizontal.3") }
                .tag(RootTab.settings)
        }
        .tint(Theme.blue)
        .environment(models)
        .environment(settings)
    }
}
