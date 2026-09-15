import SwiftUI

/// Colors and reusable surfaces. Defined in code rather than an asset catalog
/// so the palette stays readable and the xcodegen project needs no extra
/// resource wiring.
enum Theme {
    static let cyan = Color(red: 0.31, green: 0.84, blue: 0.94)
    static let blue = Color(red: 0.36, green: 0.55, blue: 0.98)
    static let violet = Color(red: 0.62, green: 0.45, blue: 0.99)
    static let pink = Color(red: 0.95, green: 0.45, blue: 0.75)
    static let mint = Color(red: 0.38, green: 0.86, blue: 0.68)
    static let amber = Color(red: 0.98, green: 0.74, blue: 0.35)
    static let danger = Color(red: 0.98, green: 0.45, blue: 0.45)

    static func gradient(for tier: Edge0Tier) -> LinearGradient {
        switch tier {
        case .edge0_8b:
            LinearGradient(
                colors: [cyan, blue], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .edge0_35b:
            LinearGradient(
                colors: [violet, pink], startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }

    static let brandGradient = LinearGradient(
        colors: [cyan, violet], startPoint: .topLeading, endPoint: .bottomTrailing)

    static let userBubble = LinearGradient(
        colors: [blue, violet], startPoint: .topLeading, endPoint: .bottomTrailing)
}

/// A rounded, subtly bordered container used for every card in the app.
struct SurfaceCard<Content: View>: View {
    var padding: CGFloat = 16
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
    }
}

/// Small pill used for metrics (tok/s, memory, token counts).
struct MetricChip: View {
    let icon: String
    let value: String
    var label: String? = nil
    var tint: Color = Theme.cyan

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .monospacedDigit()
            if let label {
                Text(label)
                    .font(.system(size: 10, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
            }
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(
            Capsule().fill(tint.opacity(0.14))
        )
    }
}

extension View {
    /// Applies the brand gradient as a foreground style — used for headings.
    func brandGradientText() -> some View {
        foregroundStyle(Theme.brandGradient)
    }
}
