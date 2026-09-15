import Combine
import SwiftUI

struct SettingsView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelManager.self) private var models
    @State private var memoryTick = Date()

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
            }
            if let loaded = models.loaded {
                LabeledContent(
                    "Parametre",
                    value: formatCount(loaded.parameterCount)
                )
                .font(.caption)
                LabeledContent("Sağlık kontrolü", value: loaded.health.detail)
                    .font(.caption)
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
            Toggle("Expert akışı (SSD offload)", isOn: settings.expertStreaming)
                .disabled(models.activeTier?.requiresExpertStreaming == true)
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
        } header: {
            Text("Çalışma zamanı")
        } footer: {
            Text(
                "35B tier'ında expert ağırlıkları bellekte tutulamaz; depolamadan akıtılır ve kapatılamaz."
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
        }
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
