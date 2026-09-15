import SwiftUI

struct ContentView: View {
    @State private var viewModel = ChatViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(viewModel.messages) { message in
                                bubble(for: message)
                                    .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.messages.count) {
                        if let last = viewModel.messages.last {
                            withAnimation {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }

                Divider()

                Text(viewModel.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 6)

                HStack(alignment: .bottom, spacing: 8) {
                    TextField("Mesajınızı yazın", text: $viewModel.input, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .disabled(viewModel.isBusy)
                        .lineLimit(1...4)

                    Button {
                        viewModel.send()
                    } label: {
                        if viewModel.isBusy {
                            ProgressView()
                                .frame(width: 28, height: 28)
                        } else {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 28))
                        }
                    }
                    .disabled(
                        viewModel.isBusy
                            || viewModel.input.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                .padding()
            }
            .navigationTitle("Edge0 Demo")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    @ViewBuilder
    private func bubble(for message: ChatMessage) -> some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }
            Text(message.text)
                .padding(10)
                .background(background(for: message.role))
                .foregroundStyle(message.role == .user ? .white : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            if message.role != .user { Spacer(minLength: 40) }
        }
    }

    private func background(for role: ChatMessage.Role) -> Color {
        switch role {
        case .user: return .blue
        case .assistant: return Color(.secondarySystemBackground)
        case .system: return Color(.tertiarySystemBackground)
        }
    }
}

#Preview {
    ContentView()
}
