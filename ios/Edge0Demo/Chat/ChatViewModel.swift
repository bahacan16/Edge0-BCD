import Foundation
import UIKit
import MLX
import MLXLMCommon
import SwiftUI

struct ChatMessage: Identifiable, Equatable, Codable {
    enum Role: String, Equatable, Codable {
        case user, assistant
    }

    var id = UUID()
    var role: Role
    var text: String
    var metrics: GenerationMetrics?
    var isStreaming: Bool = false
    var failed: Bool = false
}

struct GenerationMetrics: Equatable, Codable {
    var tokensPerSecond: Double
    var timeToFirstTokenMS: Double
    var promptTokens: Int
    var generatedTokens: Int
    var peakMemoryBytes: Int64
}

@Observable
@MainActor
final class ChatViewModel {
    var messages: [ChatMessage] = []
    var input: String = ""
    var isGenerating = false
    var errorMessage: String?

    /// Live metrics for the message currently streaming.
    var liveTokensPerSecond: Double = 0
    var liveTokenCount: Int = 0

    /// Identity of the transcript on screen, so saving updates the same file
    /// rather than piling up copies.
    private(set) var conversationID = UUID()
    private var conversationCreatedAt = Date()

    private let models: ModelManager
    private let settings: AppSettings
    private let store: ConversationStore
    private var session: ChatSession?
    private var generationTask: Task<Void, Never>?
    /// Settings that the current session was built with; a change rebuilds it.
    private var sessionSignature: String?

    init(models: ModelManager, settings: AppSettings, store: ConversationStore) {
        self.models = models
        self.settings = settings
        self.store = store
    }

    var canSend: Bool {
        (!input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !attachments.isEmpty)
            && !isGenerating
            && models.phase == .ready
    }

    // MARK: Actions

    /// Files riding along with the next turn. Cleared once it is sent — an
    /// attachment belongs to the question that was asked with it, not to the
    /// conversation, and silently re-sending a file on every later turn would
    /// make each one slower than the last for no reason anyone could see.
    var attachments: [Edge0Attachment] = []

    func attach(_ urls: [URL]) {
        for url in urls {
            do {
                let attachment = try Edge0AttachmentReader.read(url)
                guard !attachments.contains(where: { $0.name == attachment.name }) else { continue }
                attachments.append(attachment)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func removeAttachment(_ attachment: Edge0Attachment) {
        attachments.removeAll { $0.id == attachment.id }
    }

    func send() {
        let typed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let files = attachments
        guard !typed.isEmpty || !files.isEmpty, !isGenerating else { return }

        // What the model sees: the files first, then the question. What the
        // transcript shows is the same thing, so reopening a conversation later
        // shows what was actually asked rather than a question missing its
        // context.
        let block = files.promptBlock()
        let prompt = block.isEmpty ? typed : (typed.isEmpty ? block : "\(block)\n\n\(typed)")
        guard let session = currentSession() else {
            errorMessage = "Önce Modeller sekmesinden bir model yükleyin."
            return
        }

        input = ""
        attachments = []
        errorMessage = nil
        messages.append(ChatMessage(role: .user, text: prompt))
        messages.append(ChatMessage(role: .assistant, text: "", isStreaming: true))
        isGenerating = true
        liveTokensPerSecond = 0
        liveTokenCount = 0
        Haptics.tap(enabled: settings.hapticsEnabled)

        // Identity, not position: opening a saved conversation mid-generation
        // replaces `messages` wholesale, and a late chunk addressed by index
        // would be written into someone else's transcript.
        let target = messages[messages.count - 1].id
        let conversation = conversationID
        // Per turn, not per session: cumulative counters mix the load-time
        // health check and every previous answer into one figure, which is
        // exactly the figure you cannot compare between two settings.
        MLX.GPU.resetPeakMemory()
        Edge0Meter.reset()
        Edge0ExpertCaches.resetStatistics()
        // A cut made under memory pressure lasts as long as the pressure, not
        // for the rest of the session.
        Edge0ExpertCaches.restoreCapacity()

        generationTask = Task { [weak self] in
            guard let self else { return }
            let started = Date.timeIntervalSinceReferenceDate
            var firstTokenAt: TimeInterval?

            do {
                for try await generation in session.streamDetails(to: prompt) {
                    if Task.isCancelled { break }

                    if let chunk = generation.chunk, !chunk.isEmpty {
                        if firstTokenAt == nil {
                            firstTokenAt = Date.timeIntervalSinceReferenceDate
                        }
                        self.append(chunk, to: target, in: conversation, since: firstTokenAt)
                    }

                    if let info = generation.info {
                        self.finish(
                            target, in: conversation,
                            metrics: GenerationMetrics(
                                tokensPerSecond: info.tokensPerSecond,
                                timeToFirstTokenMS: ((firstTokenAt ?? started) - started) * 1000,
                                promptTokens: info.promptTokenCount,
                                generatedTokens: info.generationTokenCount,
                                peakMemoryBytes: ModelManager.mlxPeakMemoryBytes
                            ))
                    }
                }
                self.closeStream(target, in: conversation)
            } catch is CancellationError {
                // The user pressed stop; `stop()` has already tidied the
                // message up, and this is not something to show as an error.
                self.closeStream(target, in: conversation)
            } catch {
                self.fail(target, in: conversation, message: error.localizedDescription)
            }

            self.isGenerating = false
            self.generationTask = nil
            self.logGenerationStatistics(at: target, in: conversation)
            self.persist()
            Haptics.success(enabled: self.settings.hapticsEnabled)
        }
    }

    func stop() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        if let index = messages.lastIndex(where: { $0.isStreaming }) {
            messages[index].isStreaming = false
            if messages[index].text.isEmpty {
                messages[index].text = "(durduruldu)"
            }
        }
        persist()
    }

    /// Files the transcript away and starts an empty one.
    func newConversation() {
        stop()
        persist()
        messages.removeAll()
        errorMessage = nil
        conversationID = UUID()
        conversationCreatedAt = Date()
        discardSession()
    }

    /// Puts a saved transcript back on screen. The next turn rebuilds the
    /// session from it, so the model is prefilled with what was said before
    /// rather than answering out of nowhere.
    func open(_ conversation: Conversation) {
        guard conversation.id != conversationID else { return }
        stop()
        persist()
        messages = conversation.messages
        errorMessage = nil
        conversationID = conversation.id
        conversationCreatedAt = conversation.createdAt
        discardSession()
    }

    /// The conversation as shareable Markdown, or nil when there is nothing
    /// to share yet.
    var transcript: String? {
        guard !messages.isEmpty else { return nil }
        return conversation().transcript
    }

    /// Writes the transcript to disk. Called when a turn ends, not per token.
    func persist() {
        guard !messages.isEmpty else { return }
        store.save(conversation())
    }

    private func conversation() -> Conversation {
        let title = messages.first { $0.role == .user }.map { Conversation.title(from: $0.text) }
            ?? "Yeni sohbet"
        return Conversation(
            id: conversationID,
            title: title,
            tier: models.activeTier ?? settings.selectedTier,
            createdAt: conversationCreatedAt,
            updatedAt: Date(),
            messages: messages
        )
    }

    /// Called when the loaded model changes so the next turn starts clean.
    func modelChanged() {
        discardSession()
    }

    private func discardSession() {
        let session = self.session
        self.session = nil
        sessionSignature = nil
        Task { await session?.clear() }
    }

    // MARK: Session

    private func currentSession() -> ChatSession? {
        let signature = settingsSignature()
        if let session, sessionSignature == signature {
            return session
        }
        // Rebuilding — because a setting changed, or because a saved
        // transcript was opened — must not cost the model its memory of the
        // conversation, so the new session is seeded with what is on screen.
        let history = Conversation(
            title: "", tier: nil, messages: messages.filter { !$0.isStreaming }
        ).history
        guard let fresh = models.makeSession(settings: settings, history: history) else {
            return nil
        }
        session = fresh
        sessionSignature = signature
        return fresh
    }

    private func settingsSignature() -> String {
        [
            models.activeTier?.rawValue ?? "-",
            String(settings.temperature), String(settings.topP), String(settings.topK),
            String(settings.repetitionPenalty), String(settings.maxTokens),
            settings.systemPrompt, String(settings.thinkingMode),
        ].joined(separator: "|")
    }

    // MARK: Stream plumbing

    /// Speed and expert-cache behaviour, side by side in the log.
    ///
    /// On the streaming tier these two numbers explain each other: every cache
    /// miss is a read from storage in the middle of a token, so a low hit rate
    /// is what a low tokens-per-second looks like from the other end.
    private func logGenerationStatistics(at id: UUID, in conversation: UUID) {
        guard let index = index(of: id, in: conversation),
            let metrics = messages[index].metrics
        else { return }
        var line = String(
            format: "üretim: %.2f tok/sn · ilk token %.0f ms · %d token",
            metrics.tokensPerSecond, metrics.timeToFirstTokenMS, metrics.generatedTokens)
        if Edge0ExpertCaches.layerCount > 0 {
            let statistics = Edge0ExpertCaches.statistics
            let total = statistics.hits + statistics.misses
            if total > 0 {
                let percent = Double(statistics.hits) / Double(total) * 100
                line += String(
                    format: " · expert önbellek isabeti %%%.0f (%d/%d)",
                    percent, statistics.hits, total)
            }
        }
        Edge0Log.write(line)

        // The breakdown, because tokens per second moves for several reasons at
        // once and this is the only way to tell which one moved.
        Edge0Log.write(Edge0Meter.report(prerouter: settings.usePrerouter))
    }

    /// Position of the message being streamed into, or nil if the transcript
    /// it belonged to is no longer the one on screen.
    private func index(of id: UUID, in conversation: UUID) -> Int? {
        guard conversation == conversationID else { return nil }
        return messages.firstIndex { $0.id == id }
    }

    private func append(
        _ chunk: String, to id: UUID, in conversation: UUID, since firstTokenAt: TimeInterval?
    ) {
        guard let index = index(of: id, in: conversation) else { return }
        messages[index].text += chunk
        liveTokenCount += 1
        if let firstTokenAt {
            let elapsed = Date.timeIntervalSinceReferenceDate - firstTokenAt
            if elapsed > 0 {
                liveTokensPerSecond = Double(liveTokenCount) / elapsed
            }
        }
    }

    private func finish(_ id: UUID, in conversation: UUID, metrics: GenerationMetrics) {
        guard let index = index(of: id, in: conversation) else { return }
        messages[index].metrics = metrics
        messages[index].isStreaming = false
    }

    private func closeStream(_ id: UUID, in conversation: UUID) {
        guard let index = index(of: id, in: conversation) else { return }
        messages[index].isStreaming = false
        if messages[index].text.isEmpty, messages[index].metrics == nil {
            messages[index].text = "(boş yanıt)"
        }
    }

    private func fail(_ id: UUID, in conversation: UUID, message: String) {
        guard let index = index(of: id, in: conversation) else { return }
        messages[index].isStreaming = false
        messages[index].failed = true
        messages[index].text =
            messages[index].text.isEmpty ? "Hata: \(message)" : messages[index].text
        errorMessage = message
    }
}

enum Haptics {
    static func tap(enabled: Bool) {
        guard enabled else { return }
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    static func success(enabled: Bool) {
        guard enabled else { return }
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }
}
