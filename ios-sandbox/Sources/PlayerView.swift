import SwiftUI
import Lottie
import UIKit

struct LoadReport {
    var parseMs: Double = 0          // read Data + LottieAnimation decode
    var firstFrameMs: Double = 0     // load start -> first display-link tick after play
    var duration: Double = 0         // lottie-ios reported duration (s)
    var frameRate: Double = 0        // lottie-ios reported frame rate
    var frameCount: Double = 0
    var assetProbe: String = "pending"
    var failure: String?
}

struct PlayerView: View {
    let variant: Variant
    @State private var report = LoadReport()

    var body: some View {
        VStack(spacing: 0) {
            header
                .frame(height: Geometry.headerHeight)
                .frame(maxWidth: .infinity)
                .background(Color(white: 0.92))

            LottieHost(url: variant.url, report: $report)
                .frame(width: Geometry.playSide, height: Geometry.playSide)
                .background(Color.white)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Color(white: 0.92))
        .ignoresSafeArea(edges: .bottom)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(variant.displayName).font(.system(size: 15, weight: .bold, design: .monospaced))
            if let f = report.failure {
                Text("FAILURE: \(f)").font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.red)
            } else {
                Text(String(format: "dur %.3fs  fps %.1f  frames %.0f",
                            report.duration, report.frameRate, report.frameCount))
                    .font(.system(size: 12, design: .monospaced))
                Text(String(format: "parse %.0fms  firstFrame %.0fms",
                            report.parseMs, report.firstFrameMs))
                    .font(.system(size: 12, design: .monospaced))
                Text("asset: \(report.assetProbe)")
                    .font(.system(size: 11, design: .monospaced))
            }
            Text(ByteCountFormatter.string(fromByteCount: Int64(variant.byteSize),
                                           countStyle: .file))
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - UIKit host

struct LottieHost: UIViewRepresentable {
    let url: URL
    @Binding var report: LoadReport

    func makeUIView(context: Context) -> LottieHostView {
        let v = LottieHostView(frame: .zero)
        v.onReport = { r in DispatchQueue.main.async { self.report = r } }
        // `-frame N` pauses on a single frame so two variants can be diffed at
        // identical animation time; without it the view loops.
        if UserDefaults.standard.object(forKey: "frame") != nil {
            v.pinnedFrame = AnimationFrameTime(UserDefaults.standard.double(forKey: "frame"))
        }
        v.load(url: url)
        return v
    }

    func updateUIView(_ uiView: LottieHostView, context: Context) {}
}

final class LottieHostView: UIView {
    private var animationView: LottieAnimationView?
    private var displayLink: CADisplayLink?
    private var loadStart: CFAbsoluteTime = 0
    private var sawFirstFrame = false
    private var report = LoadReport()

    var onReport: ((LoadReport) -> Void)?
    var pinnedFrame: AnimationFrameTime?

    func load(url: URL) {
        loadStart = CFAbsoluteTimeGetCurrent()
        backgroundColor = .white

        // Off the main thread: read bytes, probe an embedded asset directly, and
        // decode the animation. The asset probe is independent of Lottie so a
        // webp decode failure is attributable to the image layer, not the parser.
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let data = try Data(contentsOf: url, options: .mappedIfSafe)
                let probe = Self.probeFirstEmbeddedAsset(in: data)
                let animation = try LottieAnimation.from(data: data)
                let parseMs = (CFAbsoluteTimeGetCurrent() - self.loadStart) * 1000

                DispatchQueue.main.async {
                    self.report.parseMs = parseMs
                    self.report.assetProbe = probe
                    self.report.duration = animation.duration
                    self.report.frameRate = animation.framerate
                    self.report.frameCount = animation.endFrame - animation.startFrame
                    self.install(animation)
                    self.onReport?(self.report)
                }
            } catch {
                DispatchQueue.main.async {
                    self.report.failure = String(describing: error)
                    self.onReport?(self.report)
                }
            }
        }
    }

    private func install(_ animation: LottieAnimation) {
        let av = LottieAnimationView(animation: animation)
        av.contentMode = .scaleAspectFit
        av.backgroundBehavior = .pauseAndRestore
        av.loopMode = .loop
        av.frame = bounds
        av.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        addSubview(av)
        animationView = av

        let link = CADisplayLink(target: self, selector: #selector(tick))
        link.add(to: .main, forMode: .common)
        displayLink = link

        if let f = pinnedFrame {
            av.currentFrame = f
        } else {
            av.play()
        }
    }

    @objc private func tick() {
        guard !sawFirstFrame else { return }
        // The first tick after play() is the first frame the compositor drew.
        sawFirstFrame = true
        report.firstFrameMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
        onReport?(report)
        NSLog("LOTTIE_METRIC parseMs=%.0f firstFrameMs=%.0f duration=%.3f fps=%.1f asset=%@",
              report.parseMs, report.firstFrameMs, report.duration, report.frameRate,
              report.assetProbe)
        displayLink?.invalidate()
        displayLink = nil
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        animationView?.frame = bounds
    }

    /// Finds the first `data:image/...;base64,...` payload in the raw JSON and
    /// tries to decode it with UIImage. Reports the mime type and result so a
    /// silent webp decode failure is visible as text, not just blank pixels.
    static func probeFirstEmbeddedAsset(in data: Data) -> String {
        guard let marker = "data:image/".data(using: .utf8),
              let start = data.range(of: marker)
        else { return "no data: URI found" }

        let quote = UInt8(ascii: "\"")
        guard let endIdx = data[start.lowerBound...].firstIndex(of: quote)
        else { return "unterminated data: URI" }

        guard let uri = String(data: data[start.lowerBound..<endIdx], encoding: .utf8),
              let comma = uri.firstIndex(of: ",")
        else { return "undecodable data: URI" }

        let mime = String(uri[uri.index(uri.startIndex, offsetBy: 5)..<comma])
        let b64 = String(uri[uri.index(after: comma)...])
        guard let bytes = Data(base64Encoded: b64) else { return "\(mime) base64 DECODE FAIL" }
        guard let img = UIImage(data: bytes) else { return "\(mime) UIImage FAIL (\(bytes.count)B)" }
        return String(format: "%@ OK %.0fx%.0f", mime, img.size.width, img.size.height)
    }
}
