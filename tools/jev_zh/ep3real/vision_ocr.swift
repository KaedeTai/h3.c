// Read on-screen text with Apple's Vision framework (on-device, no model to load).
// stdin: one image path per line.  stdout: one JSON object per image.
import Foundation
import Vision
import AppKit

struct Line: Codable { let text: String; let conf: Float; let x: Float; let y: Float; let w: Float; let h: Float }
struct Out: Codable { let path: String; let lines: [Line] }

func read(_ path: String) -> Out {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return Out(path: path, lines: []) }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.recognitionLanguages = ["zh-Hant", "zh-Hans", "en-US"]
    req.usesLanguageCorrection = true
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    let lines = (req.results ?? []).compactMap { o -> Line? in
        guard let c = o.topCandidates(1).first else { return nil }
        let b = o.boundingBox   // normalised, origin bottom-left
        return Line(text: c.string, conf: c.confidence, x: Float(b.minX), y: Float(1 - b.maxY), w: Float(b.width), h: Float(b.height))
    }
    return Out(path: path, lines: lines)
}

var paths: [String] = []
while let l = readLine() { if !l.isEmpty { paths.append(l) } }
var results = [Out?](repeating: nil, count: paths.count)
let lock = NSLock()
DispatchQueue.concurrentPerform(iterations: paths.count) { i in
    let r = read(paths[i]); lock.lock(); results[i] = r; lock.unlock()
}
let enc = JSONEncoder()
for r in results { if let r = r, let d = try? enc.encode(r) { print(String(data: d, encoding: .utf8)!) } }
