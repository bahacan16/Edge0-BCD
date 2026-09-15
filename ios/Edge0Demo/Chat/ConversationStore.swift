// Saved conversations.
//
// One JSON file per conversation under Application Support, written when a
// turn finishes rather than per token. Restoring one is not just cosmetic:
// `ChatSession` can be built from a message history, so the model re-prefills
// the transcript and the next turn actually remembers what was said.

import Foundation
import MLXLMCommon

struct Conversation: Codable, Identifiable, Equatable {
    var id: UUID = UUID()
    var title: String
    /// The tier the conversation was held with, for the history list. A
    /// conversation can be resumed on either tier.
    var tier: Edge0Tier?
    var createdAt: Date = Date()
    var updatedAt: Date = Date()
    var messages: [ChatMessage]

    /// Rebuilds the chat history to hand back to `ChatSession`.
    ///
    /// The chain of thought is deliberately left out: both model families are
    /// trained to see only the answer from previous turns, and replaying the
    /// reasoning would burn context on tokens the template does not expect.
    var history: [Chat.Message] {
        messages.compactMap { message in
            switch message.role {
            case .user:
                return .user(message.text)
            case .assistant:
                guard !message.failed else { return nil }
                let answer = ParsedMessage.parse(message.text).answer
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return answer.isEmpty ? nil : .assistant(answer)
            }
        }
    }

    /// The transcript as Markdown, for sharing out of the app.
    ///
    /// The chain of thought is left out for the same reason it is left out of
    /// the model's own history: it is working notes, not the answer.
    var transcript: String {
        var lines = ["# \(title)"]
        if let tier { lines.append("_\(tier.displayName) · cihaz üzerinde_") }
        lines.append("")

        for message in messages {
            switch message.role {
            case .user:
                lines.append("**Sen**")
                lines.append(message.text)
            case .assistant:
                lines.append("**Edge0**")
                lines.append(ParsedMessage.parse(message.text).answer
                    .trimmingCharacters(in: .whitespacesAndNewlines))
            }
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    static func title(from text: String) -> String {
        let cleaned = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: " ")
        guard !cleaned.isEmpty else { return "Yeni sohbet" }
        return cleaned.count > 44 ? String(cleaned.prefix(44)) + "…" : cleaned
    }
}

@Observable
@MainActor
final class ConversationStore {
    /// Newest first.
    private(set) var conversations: [Conversation] = []

    private let directory: URL
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()
    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first ?? FileManager.default.temporaryDirectory
        directory = base.appendingPathComponent("Edge0Conversations", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        reload()
    }

    func reload() {
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        conversations =
            files
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(Conversation.self, from: data)
            }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    func save(_ conversation: Conversation) {
        var conversation = conversation
        conversation.updatedAt = Date()
        // A message caught mid-stream must not come back marked as streaming.
        for index in conversation.messages.indices {
            conversation.messages[index].isStreaming = false
        }

        guard !conversation.messages.isEmpty else {
            delete(id: conversation.id)
            return
        }

        if let data = try? encoder.encode(conversation) {
            try? data.write(to: url(for: conversation.id), options: .atomic)
        }

        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = conversation
        } else {
            conversations.append(conversation)
        }
        conversations.sort { $0.updatedAt > $1.updatedAt }
    }

    func delete(id: UUID) {
        try? FileManager.default.removeItem(at: url(for: id))
        conversations.removeAll { $0.id == id }
    }

    func deleteAll() {
        for conversation in conversations {
            try? FileManager.default.removeItem(at: url(for: conversation.id))
        }
        conversations.removeAll()
    }

    private func url(for id: UUID) -> URL {
        directory.appendingPathComponent("\(id.uuidString).json")
    }
}
