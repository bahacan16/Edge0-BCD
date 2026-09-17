// UIKit's document picker, wrapped.
//
// SwiftUI's `.fileImporter` was presenting the Files browser and then never
// calling its completion: picking a file and tapping Open did nothing, and the
// log — which records the callback's first line — stayed empty across four
// different files. Whatever the reason, a modifier whose result never arrives
// cannot be debugged from the outside, and there is no way to make it say more
// than it already does not.
//
// This owns the delegate instead, so the callback is ours.
//
// `asCopy: true` is the other half of the fix. The system copies the chosen
// file into this app's temporary directory and hands back a plain local URL,
// which removes security-scoped access from the picture entirely — no
// `startAccessingSecurityScopedResource`, no coordination, no provider that
// may not have materialised the file yet. For an attachment that is read once
// and turned into text immediately, a copy is exactly the right trade.

import SwiftUI
import UniformTypeIdentifiers

struct Edge0DocumentPicker: UIViewControllerRepresentable {
    let contentTypes: [UTType]
    /// True for an attachment, which is read once and can be a copy. False for
    /// a model folder, which may be twenty gigabytes — copying that into tmp
    /// before copying it again into Models would be absurd, so those come
    /// through security-scoped and `Edge0Importer` opens the scope itself.
    var asCopy: Bool = true
    let onPick: ([URL]) -> Void
    let onCancel: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(
            forOpeningContentTypes: contentTypes, asCopy: asCopy)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = true
        picker.shouldShowFileExtensions = true
        Edge0Log.write(
            "dosya seçici açılıyor · kopya \(asCopy) · "
                + contentTypes.map(\.identifier).joined(separator: ", "))
        return picker
    }

    func updateUIViewController(_: UIDocumentPickerViewController, context: Context) {}

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        private let parent: Edge0DocumentPicker

        init(_ parent: Edge0DocumentPicker) { self.parent = parent }

        func documentPicker(
            _ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]
        ) {
            Edge0Log.write("dosya seçici: \(urls.count) dosya seçildi")
            parent.onPick(urls)
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            Edge0Log.write("dosya seçici: vazgeçildi")
            parent.onCancel()
        }
    }
}
