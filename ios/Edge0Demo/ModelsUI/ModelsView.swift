import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ModelsView: View {
    @Environment(ModelManager.self) private var models
    @Environment(AppSettings.self) private var settings
    @State private var pendingDelete: Edge0Tier?
    @State private var confirmLargeDownload: Edge0Tier?
    @State private var importingTier: Edge0Tier?
    @State private var showingImporter = false
    @State private var importError: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    ForEach(Edge0Tier.allCases) { tier in
                        TierCard(
                            tier: tier,
                            onLoad: { start(tier) },
                            onDelete: { pendingDelete = tier },
                            onImport: {
                                importingTier = tier
                                showingImporter = true
                            }
                        )
                    }

                    StorageFooter()
                    CreditsCard()
                }
                .padding(16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Modeller")
            .refreshable { models.refreshStorage() }
        }
        .onAppear { models.refreshStorage() }
        .confirmationDialog(
            "Model dosyalarını sil",
            isPresented: .init(
                get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            titleVisibility: .visible
        ) {
            Button("Sil", role: .destructive) {
                if let tier = pendingDelete { models.delete(tier: tier) }
                pendingDelete = nil
            }
            Button("Vazgeç", role: .cancel) { pendingDelete = nil }
        } message: {
            if let tier = pendingDelete {
                Text(
                    "\(tier.displayName) için indirilen \(ModelManager.formatBytes(models.diskUsage[tier] ?? 0)) silinecek."
                )
            }
        }
        .fileImporter(
            isPresented: $showingImporter,
            // Files as well as folders: picking the files inside the folder is
            // the route that works when folder-scoped access does not, and it
            // is the one the error messages point people to.
            allowedContentTypes: [.folder, .item],
            allowsMultipleSelection: true
        ) { result in
            // Falling back to the selected tier rather than returning: a lost
            // `importingTier` used to make the picker look like it did nothing.
            let tier = importingTier ?? settings.selectedTier
            importingTier = nil
            switch result {
            case .success(let urls):
                settings.selectedTier = tier
                models.importModel(tier: tier, from: urls, settings: settings)
            case .failure(let error):
                importError = error.localizedDescription
            }
        }
        .alert(
            "İçe aktarılamadı",
            isPresented: .init(
                get: { importError != nil }, set: { if !$0 { importError = nil } })
        ) {
            Button("Tamam", role: .cancel) { importError = nil }
        } message: {
            Text(importError ?? "")
        }
        .alert(
            "Büyük indirme",
            isPresented: .init(
                get: { confirmLargeDownload != nil },
                set: { if !$0 { confirmLargeDownload = nil } })
        ) {
            Button("İndir", role: .destructive) {
                if let tier = confirmLargeDownload {
                    models.prepare(tier: tier, settings: settings)
                }
                confirmLargeDownload = nil
            }
            Button("Vazgeç", role: .cancel) { confirmLargeDownload = nil }
        } message: {
            if let tier = confirmLargeDownload {
                Text(
                    """
                    \(tier.displayName) yaklaşık \(String(format: "%.0f", tier.downloadSizeGB)) GB. \
                    Wi-Fi kullanın ve indirme bitene kadar uygulamayı açık tutun.
                    """
                )
            }
        }
    }

    private func start(_ tier: Edge0Tier) {
        settings.selectedTier = tier
        if !models.downloadedTiers.contains(tier), tier.downloadSizeGB > 8 {
            confirmLargeDownload = tier
        } else {
            models.prepare(tier: tier, settings: settings)
        }
    }
}

// MARK: - Tier card

private struct TierCard: View {
    @Environment(ModelManager.self) private var models
    @Environment(AppSettings.self) private var settings
    let tier: Edge0Tier
    let onLoad: () -> Void
    let onDelete: () -> Void
    let onImport: () -> Void

    private var isActive: Bool { models.activeTier == tier }
    private var isDownloaded: Bool { models.downloadedTiers.contains(tier) }
    private var isBusyWithThis: Bool {
        models.phase.isBusy && models.pendingTier == tier
    }

    /// The load error, when this is the tier that failed. A failed phase is not
    /// a busy one, so it needs its own branch rather than living inside the
    /// progress section.
    private var failureMessage: String? {
        guard models.pendingTier == tier, case .failed(let message) = models.phase else {
            return nil
        }
        return message
    }

    var body: some View {
        SurfaceCard {
            VStack(alignment: .leading, spacing: 14) {
                header

                HStack(spacing: 8) {
                    SpecChip(icon: "square.stack.3d.up", text: "\(tier.layerCount) katman")
                    SpecChip(
                        icon: "circle.grid.3x3",
                        text: "\(tier.expertCount)×\(tier.expertsPerToken) expert")
                    SpecChip(
                        icon: "bolt", text: "~\(Int(tier.referenceTokensPerSecond)) tok/s")
                }

                if isBusyWithThis {
                    progressSection
                } else if let message = failureMessage {
                    failureSection(message)
                } else {
                    statsSection
                }

                actions
            }
        }
        .overlay(alignment: .topTrailing) {
            if isActive {
                Text("AKTİF")
                    .font(.system(size: 9, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Theme.mint))
                    .padding(10)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Theme.gradient(for: tier))
                .frame(width: 44, height: 44)
                .overlay(
                    Image(systemName: tier == .edge0_35b ? "brain" : "bolt.horizontal")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.white)
                )
            VStack(alignment: .leading, spacing: 3) {
                Text(tier.displayName)
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                Text(tier.tagline)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var memoryWarning: String {
        let available = ModelManager.formatBytes(ModelManager.availableProcessMemoryBytes)
        return "Bellek dar görünüyor (uygulamaya kalan \(available)). "
            + "Ayarlar'dan expert önbelleğini düşürmek gerekebilir."
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            LabeledRow(
                label: "İndirme boyutu",
                value: String(format: "%.1f GB", tier.downloadSizeGB))
            LabeledRow(
                label: "Tepe aktif bellek",
                value: String(format: "~%.1f GB", tier.peakActiveMemoryGB))
            if isDownloaded {
                LabeledRow(
                    label: Edge0Storage.isImported(tier) ? "İçe aktarıldı" : "Diskte",
                    value: ModelManager.formatBytes(models.diskUsage[tier] ?? 0),
                    tint: Theme.mint)
            }
            if tier.requiresExpertStreaming {
                Label(
                    "Expert ağırlıkları depolamadan akıtılır — tamamı belleğe sığmaz.",
                    systemImage: "externaldrive.connected.to.line.below"
                )
                .font(.system(size: 11))
                .foregroundStyle(Theme.amber)
            }
            if !models.hasRoom(for: tier) {
                Label("Yeterli boş alan yok.", systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Theme.danger)
            }
            if !isDownloaded {
                // Files can be dropped straight in from Finder or the Files
                // app, which skips the document picker entirely.
                Label(
                    "Dosyalar → Bu iPhone'da → \(Edge0Storage.displayPath(for: tier))",
                    systemImage: "folder"
                )
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            }
            if !models.hasMemoryHeadroom(for: tier) {
                // A warning, not a block: the budget moves around, and the
                // expert cache can be turned down to make room.
                Label(memoryWarning, systemImage: "memorychip")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.amber)
            }
        }
    }

    @ViewBuilder
    private func failureSection(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(message, systemImage: "xmark.octagon.fill")
                .font(.system(size: 11))
                .foregroundStyle(Theme.danger)
                .textSelection(.enabled)
            HStack(spacing: 8) {
                Button {
                    UIPasteboard.general.string = message
                    Haptics.tap(enabled: settings.hapticsEnabled)
                } label: {
                    Label("Hatayı kopyala", systemImage: "doc.on.doc")
                        .font(.system(size: 10, weight: .semibold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(Theme.blue)

                // LoRA is the most likely thing to go wrong and the
                // one part of the pipeline that is genuinely optional,
                // so failing with it on is worth one suggestion rather
                // than leaving retry as the only idea.
                if settings.useLoRA {
                    Button {
                        settings.useLoRA = false
                    } label: {
                        Label("LoRA'sız dene", systemImage: "wand.and.stars.inverse")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Theme.amber)
                }
            }
        }
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch models.phase {
            case .downloading(let fraction, let detail):
                ProgressView(value: fraction)
                    .tint(Theme.cyan)
                HStack {
                    Text(detail)
                    Spacer()
                    Text("%\(Int(fraction * 100))").monospacedDigit()
                }
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            case .preparing(let detail):
                ProgressView().controlSize(.small)
                Text(detail).font(.system(size: 11)).foregroundStyle(.secondary)
            default:
                EmptyView()
            }
        }
    }

    private var actions: some View {
        HStack(spacing: 10) {
            if isBusyWithThis {
                Button(role: .destructive) {
                    models.cancelLoad()
                } label: {
                    Text("Durdur").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            } else {
                Button(action: onLoad) {
                    Text(isActive ? "Yeniden yükle" : (isDownloaded ? "Yükle" : "İndir ve yükle"))
                        .font(.system(size: 14, weight: .semibold))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(tier == .edge0_35b ? Theme.violet : Theme.blue)
                .disabled(models.phase.isBusy || !models.hasRoom(for: tier))
            }

            if !isBusyWithThis {
                // A 23 GB re-download over the phone is worth avoiding when the
                // files are already sitting on a Mac, in iCloud Drive or on a
                // USB-C drive.
                Button(action: onImport) {
                    Image(systemName: "folder.badge.plus")
                }
                .buttonStyle(.bordered)
                .disabled(models.phase.isBusy)
                .accessibilityLabel("Dosyalardan içe aktar")
            }

            if isDownloaded {
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.bordered)
                .disabled(models.phase.isBusy)
            }
        }
    }
}

// MARK: - Bits

private struct SpecChip: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon).font(.system(size: 9, weight: .semibold))
            Text(text).font(.system(size: 10.5, weight: .medium, design: .rounded))
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Capsule().fill(Color.primary.opacity(0.06)))
    }
}

private struct LabeledRow: View {
    let label: String
    let value: String
    var tint: Color = .primary

    var body: some View {
        HStack {
            Text(label).font(.system(size: 12)).foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(tint)
                .monospacedDigit()
        }
    }
}

private struct StorageFooter: View {
    @Environment(ModelManager.self) private var models

    var body: some View {
        SurfaceCard(padding: 14) {
            HStack {
                Image(systemName: "internaldrive")
                    .foregroundStyle(Theme.cyan)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Boş alan")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                    Text(ModelManager.formatBytes(models.freeDiskSpace))
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("Cihaz belleği")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                    Text(ModelManager.formatBytes(ModelManager.physicalMemoryBytes))
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                }
            }
        }
    }
}

private struct CreditsCard: View {
    var body: some View {
        SurfaceCard(padding: 14) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Kaynak")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                Text(
                    """
                    Modeller ve adaptörler Edge0-AI/edge0 projesine ait (Apache-2.0). \
                    Çalıştırma Apple'ın MLX Swift kütüphanesi üzerinde yapılır.
                    """
                )
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                Link(
                    "github.com/Edge0-AI/edge0",
                    destination: URL(string: "https://github.com/Edge0-AI/edge0")!
                )
                .font(.system(size: 11.5, weight: .medium))
            }
        }
    }
}
