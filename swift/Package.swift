// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "HyperL",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(name: "HyperL", targets: ["HyperL"]),
        .executable(name: "hyperl-cli", targets: ["HyperLCLI"])
    ],
    targets: [
        .target(
            name: "CHyperL",
            path: "Sources/CHyperL",
            publicHeadersPath: "include"
        ),
        .target(
            name: "HyperL",
            dependencies: ["CHyperL"],
            path: "Sources/HyperL"
        ),
        .executableTarget(
            name: "HyperLCLI",
            dependencies: ["HyperL"],
            path: "Sources/CLI"
        )
    ]
)
