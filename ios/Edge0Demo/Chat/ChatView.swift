import SwiftUI
import UIKit

struct ChatView: View {
    @Environment(ModelManager.self) private var models
    @Environment(AppSettings.self) private var settings
    @Environment(ConversationStore.self) private var conversations
    @State private var viewModel: ChatViewModel?
    @State private var showingHistory = false
    @State private var showingAttachmentPicker = false
    @FocusState private var composerFocused: Bool
    @Binding var selectedTab: RootTab
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            Group {
                if let viewModel {
                    content(viewModel)
                } else {
                    Color(.systemGroupedBackground)
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Edge0")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbar }
        }
        .onAppear {
            if viewModel == nil {
                viewModel = ChatViewModel(
                    models: models, settings: settings, store: conversations)
            }
        }
        .onChange(of: models.activeTier) { _, _ in
            viewModel?.modelChanged()
        }
        .onChange(of: scenePhase) { _, phase in
            // Leaving the app is the one moment a half-finished transcript can
            // be lost, so it is written out even mid-answer.
            if phase != .active { viewModel?.persist() }
        }
        .sheet(isPresented: $showingHistory) {
            HistoryView(currentID: viewModel?.conversationID ?? UUID()) { conversation in
                viewModel?.open(conversation)
            }
        }
        .fileImporter(
            isPresented: $showingAttachmentPicker,
            allowedContentTypes: Edge0AttachmentReader.contentTypes,
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                Edge0Log.write("dosya seçici: \(urls.count) dosya seçildi")
                viewModel?.attach(urls)
            case .failure(let error):
                Edge0Log.failure("dosya seçici", error)
                viewModel?.errorMessage = error.localizedDescription
            }
        }
        // `errorMessage` was written in seven places and read in none, so
        // every failure below the send button — an unreadable attachment, a
        // turn started with no model loaded — happened in total silence. From
        // the outside that is indistinguishable from a button that does
        // nothing, which is exactly how it was reported.
        .alert(
            "Olmadı",
            isPresented: Binding(
                get: { viewModel?.errorMessage != nil },
                set: { if !$0 { viewModel?.errorMessage = nil } })
        ) {
            Button("Tamam", role: .cancel) { viewModel?.errorMessage = nil }
        } message: {
            Text(viewModel?.errorMessage ?? "")
        }
    }

    @ViewBuilder
    private func content(_ viewModel: ChatViewModel) -> some View {
        @Bindable var viewModel = viewModel

        VStack(spacing: 0) {
            if models.phase != .ready {
                ModelStatusBanner(selectedTab: $selectedTab)
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        if viewModel.messages.isEmpty {
                            EmptyChatState(
                                tier: models.activeTier,
                                onPick: { prompt in
                                    viewModel.input = prompt
                                    composerFocused = true
                                }
                            )
                            .padding(.top, 28)
                        }

                        ForEach(viewModel.messages) { message in
                            MessageRow(message: message, showMetrics: settings.showMetrics)
                                .id(message.id)
                        }

                        Color.clear.frame(height: 8).id(ScrollAnchor.bottom)
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: viewModel.messages.last?.text) { _, _ in
                    withAnimation(.easeOut(duration: 0.18)) {
                        proxy.scrollTo(ScrollAnchor.bottom, anchor: .bottom)
                    }
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(ScrollAnchor.bottom, anchor: .bottom)
                    }
                }
            }

            if viewModel.isGenerating, settings.showMetrics {
                LiveMetricsBar(
                    tokensPerSecond: viewModel.liveTokensPerSecond,
                    tokenCount: viewModel.liveTokenCount
                )
            }

            Composer(
                text: $viewModel.input,
                isGenerating: viewModel.isGenerating,
                canSend: viewModel.canSend,
                attachments: viewModel.attachments,
                isReading: viewModel.isReadingAttachment,
                focused: $composerFocused,
                onSend: viewModel.send,
                onStop: viewModel.stop,
                onAttach: { showingAttachmentPicker = true },
                onRemove: viewModel.removeAttachment
            )
        }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            if let tier = models.activeTier {
                Button { selectedTab = .models } label: { TierPill(tier: tier) }
                    .accessibilityLabel("Yüklü model: \(tier.displayName)")
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            if let transcript = viewModel?.transcript {
                ShareLink(item: transcript) {
                    Image(systemName: "square.and.arrow.up")
                }
                .accessibilityLabel("Sohbeti paylaş")
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button {
                showingHistory = true
            } label: {
                Image(systemName: "clock.arrow.circlepath")
            }
            .accessibilityLabel("Geçmiş")
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button {
                viewModel?.newConversation()
            } label: {
                Image(systemName: "square.and.pencil")
            }
            .disabled(viewModel?.messages.isEmpty ?? true)
            .accessibilityLabel("Yeni sohbet")
        }
    }

    private enum ScrollAnchor: Hashable {
        case bottom
    }
}

// MARK: - Message row

private struct MessageRow: View {
    let message: ChatMessage
    let showMetrics: Bool

    @State private var copied = false

    private var parsed: ParsedMessage {
        ParsedMessage.parse(message.text)
    }

    var body: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
            HStack {
                if message.role == .user { Spacer(minLength: 44) }
                bubble
                if message.role == .assistant { Spacer(minLength: 44) }
            }

            if message.role == .assistant, !message.isStreaming, !message.text.isEmpty {
                footer
            }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
        .contextMenu {
            Button {
                copy()
            } label: {
                Label("Kopyala", systemImage: "doc.on.doc")
            }
            ShareLink(item: message.text) {
                Label("Paylaş", systemImage: "square.and.arrow.up")
            }
        }
    }

    @ViewBuilder
    private var bubble: some View {
        Group {
            if message.role == .user {
                Text(message.text)
                    .textSelection(.enabled)
                    .font(.system(size: 15.5))
                    .foregroundStyle(Color.white)
            } else {
                HStack(alignment: .bottom, spacing: 3) {
                    MessageBodyView(parsed: parsed, isStreaming: message.isStreaming)
                        .textSelection(.enabled)
                        .foregroundStyle(Color.primary)
                    if message.isStreaming {
                        StreamingCaret()
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            if message.role == .user {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Theme.userBubble)
            } else {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .stroke(
                                message.failed
                                    ? Theme.danger.opacity(0.5) : Color.primary.opacity(0.06),
                                lineWidth: 1)
                    )
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Button(action: copy) {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(copied ? Theme.mint : .secondary)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(Color.primary.opacity(0.06)))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Yanıtı kopyala")

            if showMetrics, let metrics = message.metrics {
                MetricChip(
                    icon: "speedometer",
                    value: String(format: "%.1f", metrics.tokensPerSecond), label: "tok/s",
                    tint: Theme.mint)
                MetricChip(
                    icon: "timer",
                    value: String(format: "%.0f", metrics.timeToFirstTokenMS), label: "ms",
                    tint: Theme.amber)
                MetricChip(
                    icon: "number", value: "\(metrics.generatedTokens)", label: "token",
                    tint: Theme.cyan)
                MetricChip(
                    icon: "memorychip",
                    value: ModelManager.formatBytes(metrics.peakMemoryBytes),
                    tint: Theme.violet)
            }
        }
        .padding(.leading, 2)
    }

    private func copy() {
        UIPasteboard.general.string = message.text
        copied = true
        Task {
            try? await Task.sleep(for: .seconds(1.6))
            copied = false
        }
    }
}

/// The blinking block that marks where the next token will land.
private struct StreamingCaret: View {
    @State private var on = true

    var body: some View {
        RoundedRectangle(cornerRadius: 1, style: .continuous)
            .fill(Theme.cyan)
            .frame(width: 7, height: 15)
            .opacity(on ? 1 : 0.15)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.55).repeatForever(autoreverses: true)) {
                    on = false
                }
            }
    }
}

// MARK: - Composer

private struct Composer: View {
    @Binding var text: String
    let isGenerating: Bool
    let canSend: Bool
    let attachments: [Edge0Attachment]
    let isReading: Bool
    @FocusState.Binding var focused: Bool
    let onSend: () -> Void
    let onStop: () -> Void
    let onAttach: () -> Void
    let onRemove: (Edge0Attachment) -> Void

    var body: some View {
        VStack(spacing: 8) {
            if !attachments.isEmpty || isReading {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(attachments) { attachment in
                            AttachmentChip(attachment: attachment) { onRemove(attachment) }
                        }
                        if isReading {
                            HStack(spacing: 6) {
                                ProgressView().controlSize(.mini)
                                Text("okunuyor…")
                                    .font(.system(size: 12))
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                        }
                    }
                    .padding(.horizontal, 14)
                }
            }
            composer
        }
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            Button(action: onAttach) {
                Image(systemName: "paperclip")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 38, height: 38)
            }
            .disabled(isGenerating)
            .accessibilityLabel("Dosya ekle")

            TextField("Bir şey sor…", text: $text, axis: .vertical)
                .lineLimit(1...6)
                .font(.system(size: 16))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .fill(Color(.secondarySystemGroupedBackground))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                )
                .focused($focused)
                // Without this there is no way back from the keyboard on a
                // multi-line field: return inserts a newline, and the send
                // button is the only other target — so asking a question and
                // then wanting to read the answer meant being stuck behind the
                // keyboard.
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Bitti") { focused = false }
                    }
                }

            Button(action: isGenerating ? onStop : onSend) {
                Image(systemName: isGenerating ? "stop.fill" : "arrow.up")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 38, height: 38)
                    .background(Circle().fill(buttonStyle))
            }
            .disabled(!isGenerating && !canSend)
            .animation(.easeInOut(duration: 0.15), value: isGenerating)
        }
        .padding(.horizontal, 14)
    }

    private var buttonStyle: AnyShapeStyle {
        if isGenerating { return AnyShapeStyle(Theme.danger) }
        return canSend
            ? AnyShapeStyle(Theme.brandGradient) : AnyShapeStyle(Color.gray.opacity(0.45))
    }
}

// MARK: - Supporting views

private struct LiveMetricsBar: View {
    let tokensPerSecond: Double
    let tokenCount: Int

    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.mini)
            Text("üretiliyor")
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
            Spacer()
            MetricChip(
                icon: "speedometer", value: String(format: "%.1f", tokensPerSecond),
                label: "tok/s", tint: Theme.mint)
            MetricChip(icon: "number", value: "\(tokenCount)", tint: Theme.cyan)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 6)
    }
}

private struct AttachmentChip: View {
    let attachment: Edge0Attachment
    let onRemove: () -> Void

    private var icon: String {
        switch attachment.kind {
        case .text: "doc.text"
        case .drawing: "ruler"
        case .pdf: "doc.richtext"
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.cyan)
            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.name)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                Text(attachment.summary)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
            .accessibilityLabel("\(attachment.name) dosyasını kaldır")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
        .frame(maxWidth: 220)
    }
}

private struct TierPill: View {
    let tier: Edge0Tier

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(Theme.gradient(for: tier)).frame(width: 7, height: 7)
            // The short name, because the long one does not fit beside a title
            // and three buttons: it was being truncated to a lone "E", which
            // reads as a mystery button rather than as the loaded model.
            Text(tier.shortName)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .fixedSize()
        }
        .foregroundStyle(.primary)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Capsule().fill(Color.primary.opacity(0.07)))
    }
}

private struct ModelStatusBanner: View {
    @Environment(ModelManager.self) private var models
    @Environment(AppSettings.self) private var settings
    @Binding var selectedTab: RootTab

    /// The tier the user would get by tapping the button, if it is already on
    /// disk — then loading needs no trip to the models tab.
    private var readyToLoad: Edge0Tier? {
        let tier = models.activeTier ?? settings.selectedTier
        return models.downloadedTiers.contains(tier) ? tier : nil
    }

    var body: some View {
        SurfaceCard(padding: 14) {
            HStack(spacing: 12) {
                icon
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                    Text(subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer(minLength: 8)
                if case .downloading(let fraction, _) = models.phase {
                    Text("%\(Int(fraction * 100))")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(Theme.cyan)
                } else if !models.phase.isBusy {
                    Button(readyToLoad == nil ? "Aç" : "Yükle") {
                        if let tier = readyToLoad {
                            models.prepare(tier: tier, settings: settings)
                        } else {
                            selectedTab = .models
                        }
                    }
                    .font(.system(size: 13, weight: .semibold))
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.blue)
                }
            }
        }
    }

    @ViewBuilder
    private var icon: some View {
        switch models.phase {
        case .downloading, .preparing:
            ProgressView().controlSize(.small).frame(width: 26)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.danger)
                .frame(width: 26)
        default:
            Image(systemName: "cpu")
                .foregroundStyle(Theme.cyan)
                .frame(width: 26)
        }
    }

    private var title: String {
        switch models.phase {
        case .downloading: "Model indiriliyor"
        case .preparing: "Model hazırlanıyor"
        case .failed: "Model yüklenemedi"
        default: "Model yüklü değil"
        }
    }

    private var subtitle: String {
        switch models.phase {
        case .downloading(_, let detail): detail
        case .preparing(let detail): detail
        case .failed(let message): message
        default:
            readyToLoad.map { "\($0.displayName) indirilmiş — yüklemek için dokun." }
                ?? "Sohbete başlamak için Modeller sekmesinden bir tier seçin."
        }
    }
}

private struct EmptyChatState: View {
    let tier: Edge0Tier?
    let onPick: (String) -> Void

    private let suggestions = [
        "Kendini kısaca tanıt.",
        "Bir iPhone uygulaması fikri öner.",
        "Şu kodu açıkla: for i in 0..<n { sum += i }",
    ]

    var body: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle()
                    .fill(Theme.brandGradient)
                    .frame(width: 66, height: 66)
                    .blur(radius: 18)
                    .opacity(0.55)
                Image(systemName: "sparkles")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(Theme.brandGradient)
            }

            VStack(spacing: 6) {
                Text("Cihaz üzerinde çalışıyor")
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                Text(
                    tier.map { "\($0.displayName) · \($0.tagline)" }
                        ?? "Başlamak için bir model yükleyin"
                )
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            }

            VStack(spacing: 8) {
                ForEach(suggestions, id: \.self) { suggestion in
                    Button {
                        onPick(suggestion)
                    } label: {
                        HStack {
                            Text(suggestion)
                                .font(.system(size: 13.5))
                                .multilineTextAlignment(.leading)
                            Spacer(minLength: 6)
                            Image(systemName: "arrow.up.left")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 11)
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(Color(.secondarySystemGroupedBackground))
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: .infinity)
    }
}
