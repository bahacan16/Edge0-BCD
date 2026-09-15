import Foundation
import UIKit
import MLX
import MLXLMCommon
import SwiftUI

struct ChatMessage: Identifiable, Equatable {
    enum Role: Equatable {
        case user, assistant
    }

    let id = UUID()
    let role: Role
    var text: String
    var metrics: GenerationMetrics?
    var isStreaming: Bool = false
    var failed: Bool = false
}

struct GenerationMetrics: Equatable {
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

    private let models: ModelManager
    private let settings: AppSettings
    private var session: ChatSession?
    private var generationTask: Task<Void, Never>?
    /// Settings that the current session was built with; a change rebuilds it.
    private var sessionSignature: String?

    init(models: ModelManager, settings: AppSettings) {
        self.models = models
        self.settings = settings
    }

    var canSend: Bool {
        !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !isGenerating
            && models.phase == .ready
    }

    // MARK: Actions

    func send() {
        let prompt = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isGenerating else { return }
        guard let session = currentSession() else {
            errorMessage = "Önce Modeller sekmesinden bir model yükleyin."
            return
        }

        input = ""
        errorMessage = nil
        messages.append(ChatMessage(role: .user, text: prompt))
        messages.append(ChatMessage(role: .assistant, text: "", isStreaming: true))
        isGenerating = true
        liveTokensPerSecond = 0
        liveTokenCount = 0
        Haptics.tap(enabled: settings.hapticsEnabled)

        let index = messages.count - 1
        MLX.GPU.resetPeakMemory()

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
                        self.append(chunk, at: index, since: firstTokenAt)
                    }

                    if let info = generation.info {
                        self.finish(
                            at: index,
                            metrics: GenerationMetrics(
                                tokensPerSecond: info.tokensPerSecond,
                                timeToFirstTokenMS: ((firstTokenAt ?? started) - started) * 1000,
                                promptTokens: info.promptTokenCount,
                                generatedTokens: info.generationTokenCount,
                                peakMemoryBytes: ModelManager.mlxPeakMemoryBytes
                            ))
                    }
                }
                self.closeStream(at: index)
            } catch {
                self.fail(at: index, message: error.localizedDescription)
            }

            self.isGenerating = false
            self.generationTask = nil
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
    }

    /// Clears the transcript and the model's KV cache.
    func newConversation() {
        stop()
        messages.removeAll()
        errorMessage = nil
        let session = self.session
        self.session = nil
        sessionSignature = nil
        Task { await session?.clear() }
    }

    /// Called when the loaded model changes so the next turn starts clean.
    func modelChanged() {
        session = nil
        sessionSignature = nil
    }

    // MARK: Session

    private func currentSession() -> ChatSession? {
        let signature = settingsSignature()
        if let session, sessionSignature == signature {
            return session
        }
        guard let fresh = models.makeSession(settings: settings) else { return nil }
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

    private func append(_ chunk: String, at index: Int, since firstTokenAt: TimeInterval?) {
        guard messages.indices.contains(index) else { return }
        messages[index].text += chunk
        liveTokenCount += 1
        if let firstTokenAt {
            let elapsed = Date.timeIntervalSinceReferenceDate - firstTokenAt
            if elapsed > 0 {
                liveTokensPerSecond = Double(liveTokenCount) / elapsed
            }
        }
    }

    private func finish(at index: Int, metrics: GenerationMetrics) {
        guard messages.indices.contains(index) else { return }
        messages[index].metrics = metrics
        messages[index].isStreaming = false
    }

    private func closeStream(at index: Int) {
        guard messages.indices.contains(index) else { return }
        messages[index].isStreaming = false
        if messages[index].text.isEmpty, messages[index].metrics == nil {
            messages[index].text = "(boş yanıt)"
        }
    }

    private func fail(at index: Int, message: String) {
        guard messages.indices.contains(index) else { return }
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
