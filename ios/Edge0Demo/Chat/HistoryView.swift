// The saved-conversations sheet.

import SwiftUI

struct HistoryView: View {
    @Environment(ConversationStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    let currentID: UUID
    let onOpen: (Conversation) -> Void

    @State private var confirmingClearAll = false

    var body: some View {
        NavigationStack {
            Group {
                if store.conversations.isEmpty {
                    ContentUnavailableView(
                        "Henüz sohbet yok",
                        systemImage: "clock.arrow.circlepath",
                        description: Text(
                            "Bir yanıt tamamlandığında sohbet otomatik kaydedilir.")
                    )
                } else {
                    list
                }
            }
            .navigationTitle("Geçmiş")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Kapat") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button(role: .destructive) {
                        confirmingClearAll = true
                    } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(store.conversations.isEmpty)
                }
            }
            .confirmationDialog(
                "Tüm sohbetler silinsin mi?", isPresented: $confirmingClearAll,
                titleVisibility: .visible
            ) {
                Button("Hepsini sil", role: .destructive) { store.deleteAll() }
                Button("Vazgeç", role: .cancel) {}
            }
        }
    }

    private var list: some View {
        List {
            ForEach(store.conversations) { conversation in
                Button {
                    onOpen(conversation)
                    dismiss()
                } label: {
                    row(conversation)
                }
                .buttonStyle(.plain)
                .swipeActions(edge: .trailing) {
                    Button(role: .destructive) {
                        store.delete(id: conversation.id)
                    } label: {
                        Label("Sil", systemImage: "trash")
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
    }

    private func row(_ conversation: Conversation) -> some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(conversation.tier.map(Theme.gradient(for:)) ?? Theme.brandGradient)
                .frame(width: 4, height: 34)

            VStack(alignment: .leading, spacing: 3) {
                Text(conversation.title)
                    .font(.system(size: 15, weight: .medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    Text(conversation.updatedAt, format: .relative(presentation: .named))
                    Text("·")
                    Text("\(conversation.messages.count) mesaj")
                    if let tier = conversation.tier {
                        Text("·")
                        Text(tier.displayName)
                    }
                }
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                .lineLimit(1)
            }

            Spacer(minLength: 4)

            if conversation.id == currentID {
                Text("açık")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.mint)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Theme.mint.opacity(0.16)))
            }
        }
        .contentShape(Rectangle())
    }
}
