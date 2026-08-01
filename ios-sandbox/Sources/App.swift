import SwiftUI

/// Fixed geometry so screenshot analysis knows exactly where the play area is.
/// Header occupies the top `headerHeight` points; the play area is a square of
/// `playSide` points, horizontally centred, starting immediately below.
enum Geometry {
    static let headerHeight: CGFloat = 140
    static let playSide: CGFloat = 320
}

struct Variant: Identifiable, Hashable {
    let id: String        // file basename without extension
    var url: URL
    var byteSize: Int

    var displayName: String { id }
}

enum VariantCatalog {
    /// Ordered so the lossless PNG control comes first for each subject.
    static let preferredOrder = ["512", "512q", "512webp", "256webp"]

    static func discover() -> [Variant] {
        guard let dir = Bundle.main.url(forResource: "lotties", withExtension: nil),
              let files = try? FileManager.default.contentsOfDirectory(
                  at: dir, includingPropertiesForKeys: [.fileSizeKey])
        else { return [] }

        let variants: [Variant] = files
            .filter { $0.pathExtension == "json" }
            .map { url in
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                return Variant(id: url.deletingPathExtension().lastPathComponent,
                               url: url, byteSize: size)
            }

        return variants.sorted { a, b in
            let (sa, ra) = split(a.id), (sb, rb) = split(b.id)
            if sa != sb { return sa < sb }
            return rank(ra) < rank(rb)
        }
    }

    private static func split(_ id: String) -> (String, String) {
        guard let dot = id.lastIndex(of: ".") else { return (id, "") }
        return (String(id[id.startIndex..<dot]), String(id[id.index(after: dot)...]))
    }

    private static func rank(_ rung: String) -> Int {
        preferredOrder.firstIndex(of: rung) ?? preferredOrder.count
    }
}

@main
struct LottieSandboxApp: App {
    /// Set by `xcrun simctl launch <sim> <bundle> -variant <name>`, which lands
    /// in UserDefaults. When present the app boots straight into the player so
    /// verification needs no UI automation.
    private var autoVariant: Variant? {
        guard let name = UserDefaults.standard.string(forKey: "variant") else { return nil }
        return VariantCatalog.discover().first { $0.id == name }
    }

    var body: some Scene {
        WindowGroup {
            if let v = autoVariant {
                PlayerView(variant: v)
            } else {
                VariantListView()
            }
        }
    }
}

struct VariantListView: View {
    private let variants = VariantCatalog.discover()

    var body: some View {
        NavigationStack {
            List(variants) { v in
                NavigationLink(value: v) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(v.displayName).font(.system(.body, design: .monospaced))
                        Text(ByteCountFormatter.string(fromByteCount: Int64(v.byteSize),
                                                       countStyle: .file))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Lottie Sandbox")
            .navigationDestination(for: Variant.self) { PlayerView(variant: $0) }
        }
    }
}
