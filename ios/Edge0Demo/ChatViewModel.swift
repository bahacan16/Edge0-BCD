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
                Edge0-8B (Edge0/Edge0-8B-A1B-preview): Edge0-AI/edge0'ın Ling 3.0 \
                tabanlı hibrit modeli, cihaz üzerinde çalışıyor. İlk mesajınızda \
                model + LoRA adaptörü Hugging Face'ten indirilir (Wi-Fi önerilir, \
                ~4-5 GB), sonrasında tamamen telefonda, internete gitmeden üretim \
                yapılır. SSD expert-offload/prerouter hızlandırması yok — model \
                tamamen bellekte tutuluyor.
                """
        )
    ]
    var input: String = ""
    var isBusy = false
    var statusText = "Hazır"

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

        statusText = "Edge0-8B indiriliyor (ilk çalıştırma uzun sürebilir, ~4-5 GB)…"
        let model = try await Edge0Model.load { [weak self] progress in
            Task { @MainActor in
                let pct = Int(progress.fractionCompleted * 100)
                self?.statusText = "Edge0-8B indiriliyor… %\(pct)"
            }
        }
        statusText = "Model yükleniyor…"
        let newSession = ChatSession(model)
        session = newSession
        return newSession
    }
}
