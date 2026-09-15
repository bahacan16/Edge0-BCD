import SwiftUI

enum RootTab: Hashable {
    case chat, models, settings
}

struct RootView: View {
    @State private var models = ModelManager()
    @State private var settings = AppSettings.shared
    @State private var conversations = ConversationStore()
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
        .environment(conversations)
        .task {
            // Reading a checkpoint back off disk takes long enough that having
            // to ask for it every launch is a chore. Only ever loads something
            // already downloaded — this never starts a download.
            guard settings.autoLoadLastModel, models.phase == .idle, models.loaded == nil,
                models.downloadedTiers.contains(settings.selectedTier)
            else { return }
            models.prepare(tier: settings.selectedTier, settings: settings)
        }
    }
}
