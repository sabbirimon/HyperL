# Bundled workbench font provenance

The UI polish increment bundles three unmodified static font files as application
resources. They are loaded locally with Java's font API; no OS install or runtime
network download is required. No font was extracted from the owner's screenshots.
Both families retain their upstream SIL Open Font License 1.1/copyright texts in
`docs/licenses`, included in ZIP/TAR distributions. HyperL's source license does
not replace the font licenses. Logical system-font fallback is used if loading fails.

| Resource | Pinned source | SHA-256 |
|---|---|---|
| Inter-Regular.otf | FlatLaf tag 3.7.2, `flatlaf-fonts/flatlaf-fonts-inter/src/main/resources/com/formdev/flatlaf/fonts/inter/Inter-Regular.otf` | `d4f2b9e148059a15f014cb0f0b8fea8cd11bfa447dd483bedf1b0adc0e2ba799` |
| Inter-SemiBold.otf | Same tag/directory, `Inter-SemiBold.otf` | `0a4d30778fa2dc239368d90fd854b88b27e7ede737450da72686a90eca66f02c` |
| JetBrainsMono-Regular.ttf | JetBrains/JetBrainsMono tag v2.304, `fonts/ttf/JetBrainsMono-Regular.ttf` | `a0bf60ef0f83c5ed4d7a75d45838548b1f6873372dfac88f71804491898d138f` |

Verified downloaded content against pinned upstream Git blob identities before
adding it. Total raw font resources: **1,513,412 bytes**. The GUI action checks also
verify the actual Inter and JetBrains Mono families loaded. Native display/HiDPI/
accessibility remains a separate OS session qualification, not inferred from fonts.

Sources: [FlatLaf pinned Inter resources](https://github.com/JFormDesigner/FlatLaf/tree/3.7.2/flatlaf-fonts/flatlaf-fonts-inter/src/main/resources/com/formdev/flatlaf/fonts/inter),
[Inter project](https://rsms.me/inter/),
[JetBrains Mono v2.304](https://github.com/JetBrains/JetBrainsMono/tree/v2.304),
[JetBrains Mono license description](https://www.jetbrains.com/lp/mono/).
