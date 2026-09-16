import Combine
import SwiftUI
import UIKit

struct SettingsView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelManager.self) private var models
    @State private var memoryTick = Date()
    @State private var copiedDiagnostics = false

    private let timer = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        @Bindable var settings = settings

        NavigationStack {
            Form {
                modelSection($settings)
                samplingSection($settings)
                conversationSection($settings)
                runtimeSection($settings)
                interfaceSection($settings)
                aboutSection
            }
            .navigationTitle("Ayarlar")
        }
        .onReceive(timer) { _ in memoryTick = Date() }
    }

    // MARK: Sections

    @ViewBuilder
    private func modelSection(_ settings: Bindable<AppSettings>) -> some View {
        Section {
            HStack {
                Text("Aktif model")
                Spacer()
                Text(models.activeTier?.displayName ?? "Yok")
                    .foregroundStyle(.secondary)
            }
            Toggle("Açılışta son modeli yükle", isOn: settings.autoLoadLastModel)
            Toggle("Recover-LoRA adaptörleri", isOn: settings.useLoRA)
            if models.activeTier != nil {
                Text(
                    "LoRA değişikliği bir sonraki model yüklemesinde geçerli olur."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            if let report = models.loaded?.loraReport {
                LabeledContent("Uygulanan adaptör", value: "\(report.appliedTargets.count)")
                    .font(.caption)
                if !report.unmatchedTargets.isEmpty {
                    // Silently dropped adapters would look like a plain quality
                    // regression, so the count is surfaced rather than logged.
                    LabeledContent(
                        "Eşleşmeyen adaptör", value: "\(report.unmatchedTargets.count)"
                    )
                    .font(.caption)
                    .foregroundStyle(Theme.amber)
                }
            }
            if let loaded = models.loaded {
                LabeledContent(
                    "Parametre",
                    value: formatCount(loaded.parameterCount)
                )
                .font(.caption)
                LabeledContent("Sağlık kontrolü", value: loaded.health.detail)
                    .font(.caption)
                if !loaded.health.sample.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Açılış örneği")
                            .font(.caption)
                        Text(loaded.health.sample)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                }
            }
        } header: {
            Text("Model")
        } footer: {
            Text(
                "edge0, 4-bit tabanı dondurup LoRA adaptörlerini ayrı tutar; kapatmak kaliteyi düşürür."
            )
        }
    }

    @ViewBuilder
    private func samplingSection(_ settings: Bindable<AppSettings>) -> some View {
        Section {
            SliderRow(
                title: "Sıcaklık", value: settings.temperature, range: 0...1.5, step: 0.05,
                format: "%.2f", tint: Theme.amber)
            SliderRow(
                title: "Top-p", value: settings.topP, range: 0.1...1.0, step: 0.01,
                format: "%.2f", tint: Theme.cyan)
            StepperRow(
                title: "Top-k", value: settings.topK, range: 0...200, step: 8,
                caption: settings.wrappedValue.topK == 0 ? "kapalı" : nil)
            SliderRow(
                title: "Tekrar cezası", value: settings.repetitionPenalty, range: 1.0...1.5,
                step: 0.01, format: "%.2f", tint: Theme.violet,
                caption: settings.wrappedValue.repetitionPenalty <= 1.0 ? "kapalı" : nil)
            StepperRow(
                title: "Maks. token", value: settings.maxTokens, range: 128...8192, step: 128)

            Button("Tier varsayılanlarına dön") {
                settings.wrappedValue.resetSamplingToTierDefaults()
            }
        } header: {
            Text("Üretim")
        } footer: {
            Text("Değişiklikler bir sonraki mesajda uygulanır.")
        }
    }

    @ViewBuilder
    private func conversationSection(_ settings: Bindable<AppSettings>) -> some View {
        Section("Sohbet") {
            VStack(alignment: .leading, spacing: 6) {
                Text("Sistem istemi")
                    .font(.system(size: 13, weight: .medium))
                TextField(
                    "Sistem istemi", text: settings.systemPrompt, axis: .vertical
                )
                .lineLimit(2...6)
                .font(.system(size: 13))
                .textFieldStyle(.plain)
            }
            Toggle("Düşünme modu", isOn: settings.thinkingMode)
        }
    }

    @ViewBuilder
    private func runtimeSection(_ settings: Bindable<AppSettings>) -> some View {
        Section {
            StepperRow(
                title: "MLX önbellek sınırı", value: settings.gpuCacheLimitMB,
                range: 0...1024, step: 32, unit: "MB")
            // A tier that requires streaming gets it whatever the switch says,
            // so the switch has to read as on rather than quietly contradict
            // what the app is doing.
            Toggle(
                "Expert akışı (SSD offload)",
                isOn: (models.activeTier ?? settings.wrappedValue.selectedTier)
                    .requiresExpertStreaming
                    ? .constant(true) : settings.expertStreaming
            )
            .disabled(!(models.activeTier ?? settings.wrappedValue.selectedTier)
                .supportsExpertStreaming
                || (models.activeTier ?? settings.wrappedValue.selectedTier)
                    .requiresExpertStreaming)
            StepperRow(
                title: "Expert önbelleği", value: settings.expertCacheBudgetMB,
                range: 256...4096, step: 256, unit: "MB")
            let prerouterTier = models.activeTier ?? settings.wrappedValue.selectedTier
            Toggle("Prerouter (bir adım önden okuma)", isOn: settings.usePrerouter)
                .disabled(prerouterTier.prerouterFileName == nil)
            Text(
                prerouterTier.prerouterFileName == nil
                    ? "Bu katman için prerouter portu yok; kapılarla çalışır."
                    : "Her katman bir sonrakinin yönlendirmesini bir token önceden"
                        + " tahmin eder, böylece bir adımın bütün expert'leri"
                        + " diskten aynı anda okunur. Recover-LoRA da bu kurulum"
                        + " için eğitildi. Değişiklik bir sonraki yüklemede geçerli olur."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            LabeledContent("MLX aktif bellek") {
                Text(ModelManager.formatBytes(ModelManager.mlxActiveMemoryBytes))
                    .monospacedDigit()
            }
            LabeledContent("MLX tepe bellek") {
                Text(ModelManager.formatBytes(ModelManager.mlxPeakMemoryBytes))
                    .monospacedDigit()
            }
            LabeledContent("Cihaz belleği") {
                Text(ModelManager.formatBytes(ModelManager.physicalMemoryBytes))
                    .monospacedDigit()
            }
            LabeledContent("Uygulamaya kalan") {
                Text(ModelManager.formatBytes(ModelManager.availableProcessMemoryBytes))
                    .monospacedDigit()
                    .foregroundStyle(Theme.mint)
            }
            if Edge0ExpertCaches.layerCount > 0 {
                let statistics = Edge0ExpertCaches.statistics
                LabeledContent("Expert önbellek isabeti") {
                    Text(hitRate(statistics))
                        .monospacedDigit()
                }
                .font(.caption)
            }
        } header: {
            Text("Çalışma zamanı")
        } footer: {
            Text(
                """
                35B tier'ında expert ağırlıkları bellekte tutulamaz; depolamadan akıtılır ve \
                kapatılamaz. Önbellek bütçesi tüm MoE katmanları arasında paylaşılır: büyüttükçe \
                uzun promptlar hızlanır, bellek kullanımı artar. Cihaz bellek uyarısı verirse \
                önbellek otomatik boşaltılır.
                """
            )
        }
        .id(memoryTick)
    }

    @ViewBuilder
    private func interfaceSection(_ settings: Bindable<AppSettings>) -> some View {
        Section("Arayüz") {
            Toggle("Ölçümleri göster", isOn: settings.showMetrics)
            Toggle("Titreşim", isOn: settings.hapticsEnabled)
        }
    }

    private var aboutSection: some View {
        Section("Hakkında") {
            LabeledContent("Model kaynağı", value: "Edge0-AI/edge0")
            LabeledContent("Çalışma zamanı", value: "MLX Swift")
            Link(
                "Edge0 deposu",
                destination: URL(string: "https://github.com/Edge0-AI/edge0")!)
            Link(
                "MLX Swift",
                destination: URL(string: "https://github.com/ml-explore/mlx-swift")!)

            ShareLink(item: Edge0Log.fileURL) {
                Label("Günlük dosyasını paylaş", systemImage: "doc.text")
            }
            Button(role: .destructive) {
                Edge0Log.clear()
            } label: {
                Label("Günlüğü temizle", systemImage: "trash")
            }
            Text("Günlük: Dosyalar → Bu iPhone'da → Edge0 Demo → Edge0.log")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            Button {
                UIPasteboard.general.string = diagnostics
                copiedDiagnostics = true
                Task {
                    try? await Task.sleep(for: .seconds(1.8))
                    copiedDiagnostics = false
                }
            } label: {
                Label(
                    copiedDiagnostics ? "Kopyalandı" : "Tanılama bilgisini kopyala",
                    systemImage: copiedDiagnostics ? "checkmark" : "stethoscope")
            }
            .foregroundStyle(copiedDiagnostics ? Theme.mint : Theme.blue)
        }
    }

    /// Everything worth knowing when the model misbehaves, in one paste.
    ///
    /// Bad output from a hand-written port is nearly impossible to diagnose
    /// from a description of it — what settles it is which tier, whether the
    /// adapters all matched, what the load-time sample said and how much
    /// memory there was. Asking someone to read six screens back is how that
    /// never gets reported.
    private var diagnostics: String {
        var lines: [String] = ["Edge0 tanılama"]

        let version =
            Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        lines.append("Uygulama: \(version) (\(build))")
        lines.append("iOS: \(UIDevice.current.systemVersion)")
        lines.append("Tier: \(models.activeTier?.displayName ?? "yüklü değil")")
        lines.append("Faz: \(String(describing: models.phase))")

        if let loaded = models.loaded {
            lines.append("Parametre: \(formatCount(loaded.parameterCount))")
            lines.append("Sağlık: \(loaded.health.detail)")
            lines.append("Sözlük: \(loaded.health.vocabularySize)")
            if !loaded.health.sample.isEmpty {
                lines.append("Açılış örneği: \(loaded.health.sample)")
            }
            if let report = loaded.loraReport {
                lines.append(
                    "LoRA: uygulanan \(report.appliedTargets.count),"
                        + " eşleşmeyen \(report.unmatchedTargets.count),"
                        + " ölçek \(report.scale)")
                // The first few say which layer or projection went astray,
                // which is the difference between a guess and a fix.
                for target in report.unmatchedTargets.prefix(5) {
                    lines.append("  eşleşmeyen: \(target)")
                }
                if let rank = report.rankMismatch {
                    lines.append("  UYARI: adaptör rank'i \(rank), ölçek başka rank varsayıyor")
                }
            } else {
                lines.append("LoRA: uygulanmadı")
            }
        }

        lines.append("LoRA açık: \(settings.useLoRA)")
        lines.append("Expert akışı: \(settings.expertStreaming)")
        lines.append("Expert önbelleği: \(settings.expertCacheBudgetMB) MB")
        lines.append("Prerouter: \(settings.usePrerouter)")
        lines.append("MLX önbellek sınırı: \(settings.gpuCacheLimitMB) MB")
        lines.append(
            "Üretim: T=\(settings.temperature) topP=\(settings.topP)"
                + " topK=\(settings.topK) rep=\(settings.repetitionPenalty)"
                + " maks=\(settings.maxTokens) düşünme=\(settings.thinkingMode)")

        if Edge0ExpertCaches.layerCount > 0 {
            let statistics = Edge0ExpertCaches.statistics
            lines.append(
                "Akıtılan katman: \(Edge0ExpertCaches.layerCount),"
                    + " önbellek isabeti \(hitRate(statistics))")
        }

        lines.append("MLX aktif: \(ModelManager.formatBytes(ModelManager.mlxActiveMemoryBytes))")
        lines.append("MLX tepe: \(ModelManager.formatBytes(ModelManager.mlxPeakMemoryBytes))")
        lines.append(
            "Uygulamaya kalan: "
                + ModelManager.formatBytes(ModelManager.availableProcessMemoryBytes))
        lines.append("Cihaz belleği: \(ModelManager.formatBytes(ModelManager.physicalMemoryBytes))")
        lines.append("Boş disk: \(ModelManager.formatBytes(models.freeDiskSpace))")

        return lines.joined(separator: "\n")
    }

    /// Share of expert reads served from RAM rather than from storage. A low
    /// rate on long prompts is the signal that the cache budget is too small.
    private func hitRate(_ statistics: (hits: Int, misses: Int)) -> String {
        let total = statistics.hits + statistics.misses
        guard total > 0 else { return "—" }
        let percent = Double(statistics.hits) / Double(total) * 100
        return String(format: "%%%.0f (%d/%d)", percent, statistics.hits, total)
    }

    private func formatCount(_ value: Int) -> String {
        let millions = Double(value) / 1_000_000
        if millions >= 1000 {
            return String(format: "%.1fB", millions / 1000)
        }
        return String(format: "%.0fM", millions)
    }
}

// MARK: - Rows

private struct SliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    var format: String = "%.2f"
    var tint: Color = Theme.blue
    var caption: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.system(size: 14))
                Spacer()
                Text(caption ?? String(format: format, value))
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(caption == nil ? tint : .secondary)
            }
            Slider(value: $value, in: range, step: step)
                .tint(tint)
        }
        .padding(.vertical, 2)
    }
}

private struct StepperRow: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int
    var unit: String? = nil
    var caption: String? = nil

    var body: some View {
        Stepper(value: $value, in: range, step: step) {
            HStack {
                Text(title).font(.system(size: 14))
                Spacer()
                Text(caption ?? "\(value)\(unit.map { " \($0)" } ?? "")")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(caption == nil ? .primary : .secondary)
            }
        }
    }
}
