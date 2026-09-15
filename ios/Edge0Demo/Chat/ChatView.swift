import SwiftUI

struct ChatView: View {
    @Environment(ModelManager.self) private var models
    @Environment(AppSettings.self) private var settings
    @State private var viewModel: ChatViewModel?
    @FocusState private var composerFocused: Bool
    @Binding var selectedTab: RootTab

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
                viewModel = ChatViewModel(models: models, settings: settings)
            }
        }
        .onChange(of: models.activeTier) { _, _ in
            viewModel?.modelChanged()
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
                focused: $composerFocused,
                onSend: viewModel.send,
                onStop: viewModel.stop
            )
        }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            if let tier = models.activeTier {
                TierPill(tier: tier)
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button {
                viewModel?.newConversation()
            } label: {
                Image(systemName: "square.and.pencil")
            }
            .disabled(viewModel?.messages.isEmpty ?? true)
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

    var body: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
            HStack {
                if message.role == .user { Spacer(minLength: 44) }
                bubble
                if message.role == .assistant { Spacer(minLength: 44) }
            }

            if showMetrics, let metrics = message.metrics {
                HStack(spacing: 6) {
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
                .padding(.leading, 2)
            }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }

    private var bubble: some View {
        Text(displayText)
            .textSelection(.enabled)
            .font(.system(size: 15.5))
            .foregroundStyle(message.role == .user ? Color.white : Color.primary)
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

    private var displayText: String {
        if message.isStreaming && message.text.isEmpty {
            return "…"
        }
        return message.text + (message.isStreaming ? " ▍" : "")
    }
}

// MARK: - Composer

private struct Composer: View {
    @Binding var text: String
    let isGenerating: Bool
    let canSend: Bool
    @FocusState.Binding var focused: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
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
                .submitLabel(.send)

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
        .padding(.vertical, 10)
        .background(.bar)
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

private struct TierPill: View {
    let tier: Edge0Tier

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(Theme.gradient(for: tier)).frame(width: 7, height: 7)
            Text(tier.displayName)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Capsule().fill(Color.primary.opacity(0.07)))
    }
}

private struct ModelStatusBanner: View {
    @Environment(ModelManager.self) private var models
    @Binding var selectedTab: RootTab

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
                    Button("Aç") { selectedTab = .models }
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
        default: "Sohbete başlamak için Modeller sekmesinden bir tier seçin."
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
