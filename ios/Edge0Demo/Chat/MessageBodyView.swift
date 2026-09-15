// Renders a parsed answer: the collapsible chain of thought, then the blocks.

import SwiftUI
import UIKit

struct MessageBodyView: View {
    let parsed: ParsedMessage
    let isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if parsed.hasReasoning {
                ReasoningView(
                    text: parsed.reasoning,
                    isThinking: parsed.reasoningIsOpen && isStreaming)
            }

            if parsed.hasAnswer {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(parsed.blocks) { block in
                        blockView(block)
                    }
                }
            } else if !parsed.hasReasoning {
                Text(isStreaming ? "…" : "")
                    .font(.system(size: 15.5))
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: MessageBlock) -> some View {
        switch block.kind {
        case .heading(let level):
            Text(InlineMarkdown.attributed(block.text))
                .font(.system(size: level == 1 ? 19 : (level == 2 ? 17 : 16), weight: .semibold))
                .padding(.top, 2)

        case .paragraph:
            Text(InlineMarkdown.attributed(block.text))
                .font(.system(size: 15.5))
                .fixedSize(horizontal: false, vertical: true)

        case .bullet(let marker):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(marker)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.blue)
                    .monospacedDigit()
                Text(InlineMarkdown.attributed(block.text))
                    .font(.system(size: 15.5))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, 2)

        case .quote:
            HStack(alignment: .top, spacing: 10) {
                Capsule()
                    .fill(Theme.violet.opacity(0.5))
                    .frame(width: 3)
                Text(InlineMarkdown.attributed(block.text))
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .fixedSize(horizontal: false, vertical: true)

        case .rule:
            Divider().padding(.vertical, 2)

        case .code(let language):
            CodeBlockView(language: language, code: block.text)
        }
    }
}

// MARK: - Chain of thought

private struct ReasoningView: View {
    let text: String
    let isThinking: Bool

    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isThinking ? "brain.head.profile" : "text.bubble")
                        .font(.system(size: 11, weight: .semibold))
                        .symbolEffect(.pulse, isActive: isThinking)
                    Text(isThinking ? "Düşünüyor…" : "Düşünce zinciri")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                    Image(systemName: "chevron.down")
                        .font(.system(size: 9, weight: .bold))
                        .rotationEffect(.degrees(expanded ? 0 : -90))
                }
                .foregroundStyle(Theme.violet)
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(Capsule().fill(Theme.violet.opacity(0.14)))
            }
            .buttonStyle(.plain)

            if expanded {
                Text(text.trimmingCharacters(in: .whitespacesAndNewlines))
                    .font(.system(size: 13.5))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 4)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        // Reasoning can run for a while before a single word of the answer
        // appears, so it is shown while it happens and folded away once the
        // answer takes over. Tapping after that reopens it.
        .onAppear { expanded = isThinking }
        .onChange(of: isThinking) { _, nowThinking in
            guard !nowThinking else { return }
            withAnimation(.easeInOut(duration: 0.2)) { expanded = false }
        }
    }
}

// MARK: - Code

private struct CodeBlockView: View {
    let language: String?
    let code: String

    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language?.uppercased() ?? "KOD")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Button {
                    UIPasteboard.general.string = code
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.6))
                        copied = false
                    }
                } label: {
                    Label(
                        copied ? "Kopyalandı" : "Kopyala",
                        systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10, weight: .semibold))
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.plain)
                .foregroundStyle(copied ? Theme.mint : Theme.blue)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)

            Divider()

            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(size: 13, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(10)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.tertiarySystemGroupedBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.primary.opacity(0.07), lineWidth: 1)
        )
    }
}
