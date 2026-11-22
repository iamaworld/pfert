import Foundation
import Vision
import CoreImage
import CoreGraphics

// One process handles a query and all catalog photos, so the model is loaded once.
// Input: vision-match query.jpg reference1.jpg reference2.jpg ...
// Output: JSON array ordered like the reference paths.
struct Region: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct Match: Codable {
    let index: Int
    let distance: Float?
    let region: Region?
}

struct View {
    let print: VNFeaturePrintObservation
    let region: Region
}

let context = CIContext(options: [.useSoftwareRenderer: true])

func image(at path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path)
    guard let ci = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else {
        throw NSError(domain: "TakhiVision", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot open image: \(path)"])
    }
    let scale = min(1.0, 1600.0 / max(ci.extent.width, ci.extent.height))
    let prepared = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    guard let cg = context.createCGImage(prepared, from: prepared.extent) else {
        throw NSError(domain: "TakhiVision", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot decode image: \(path)"])
    }
    return cg
}

func views(for image: CGImage, query: Bool) -> [View] {
    let regions: [Region] = query ? [
        Region(x: 0, y: 0, width: 1, height: 1),
        Region(x: 0.14, y: 0.14, width: 0.72, height: 0.72),
        Region(x: 0, y: 0.10, width: 0.68, height: 0.80),
        Region(x: 0.32, y: 0.10, width: 0.68, height: 0.80),
        Region(x: 0.10, y: 0, width: 0.80, height: 0.68),
        Region(x: 0.10, y: 0.32, width: 0.80, height: 0.68),
        Region(x: 0, y: 0, width: 0.60, height: 0.60),
        Region(x: 0.40, y: 0, width: 0.60, height: 0.60),
        Region(x: 0, y: 0.40, width: 0.60, height: 0.60),
        Region(x: 0.40, y: 0.40, width: 0.60, height: 0.60)
    ] : [
        Region(x: 0, y: 0, width: 1, height: 1),
        Region(x: 0.10, y: 0.10, width: 0.80, height: 0.80)
    ]
    return regions.compactMap { region in
        let rectangle = CGRect(x: Int(Double(image.width) * region.x),
                               y: Int(Double(image.height) * region.y),
                               width: max(1, Int(Double(image.width) * region.width)),
                               height: max(1, Int(Double(image.height) * region.height)))
        guard let crop = image.cropping(to: rectangle) else { return nil }
        let request = VNGenerateImageFeaturePrintRequest()
        do {
            try VNImageRequestHandler(cgImage: crop).perform([request])
            guard let observation = request.results?.first else { return nil }
            return View(print: observation, region: region)
        } catch {
            return nil
        }
    }
}

do {
    let paths = Array(CommandLine.arguments.dropFirst())
    guard paths.count > 1 else { throw NSError(domain: "TakhiVision", code: 2) }
    let queryViews = views(for: try image(at: paths[0]), query: true)
    guard !queryViews.isEmpty else { throw NSError(domain: "TakhiVision", code: 3) }
    var matches: [Match] = []
    for (index, path) in paths.dropFirst().enumerated() {
        do {
            let referenceViews = views(for: try image(at: path), query: false)
            var best: Float = .greatestFiniteMagnitude
            var bestRegion: Region?
            for queryView in queryViews {
                for referenceView in referenceViews {
                    var distance: Float = 0
                    try queryView.print.computeDistance(&distance, to: referenceView.print)
                    if distance < best {
                        best = distance
                        bestRegion = queryView.region
                    }
                }
            }
            matches.append(Match(index: index, distance: best.isFinite ? best : nil, region: bestRegion))
        } catch {
            matches.append(Match(index: index, distance: nil, region: nil))
        }
    }
    let output = try JSONEncoder().encode(matches)
    FileHandle.standardOutput.write(output)
} catch {
    fputs("Vision failed: \(error)\n", stderr)
    exit(1)
}
