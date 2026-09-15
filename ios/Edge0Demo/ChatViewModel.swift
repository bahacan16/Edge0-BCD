import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@Observable
@MainActor
final class ChatViewModel {
    var messages: [ChatMessage] = [
        ChatMessage(
            role: .system,
            text: """
                Edge0 Demo: cihaz üzerinde (on-device) çalışan küçük bir dil modeli.
                İlk mesajınızı gönderdiğinizde model Hugging Face'ten indirilir \
                (Wi-Fi önerilir, birkaç yüz MB) ve sonraki tüm üretim tamamen \
                telefonda, internete gitmeden çalışır.
                """
        )
    ]
    var input: String = ""
    var isBusy = false
    var statusText = "Hazır"

    private let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit
    private var session: ChatSession?

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isBusy else { return }

        input = ""
        messages.append(ChatMessage(role: .user, text: text))
        isBusy = true

        Task {
            do {
                let session = try await loadSessionIfNeeded()
                statusText = "Yanıt üretiliyor…"
                let reply = try await session.respond(to: text)
                messages.append(ChatMessage(role: .assistant, text: reply))
                statusText = "Hazır"
            } catch {
                messages.append(
                    ChatMessage(role: .assistant, text: "Hata: \(error.localizedDescription)"))
                statusText = "Hata oluştu"
            }
            isBusy = false
        }
    }

    private func loadSessionIfNeeded() async throws -> ChatSession {
        if let session {
            return session
        }

        statusText = "Model indiriliyor (ilk çalıştırma birkaç dakika sürebilir)…"
        let model = try await #huggingFaceLoadModelContainer(configuration: modelConfiguration)
        let newSession = ChatSession(model)
        session = newSession
        return newSession
    }
}
