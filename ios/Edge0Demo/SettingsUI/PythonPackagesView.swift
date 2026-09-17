// Package management, the part of pip worth having on a phone.

import SwiftUI

struct PythonPackagesView: View {
    @State private var store = Edge0PythonPackages()
    @State private var requirement = ""
    @FocusState private var fieldFocused: Bool

    var body: some View {
        Form {
            Section("Paket kur") {
                HStack {
                    TextField("ezdxf ya da ezdxf==1.0.3", text: $requirement)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .focused($fieldFocused)
                        .onSubmit(begin)
                    if store.busy {
                        ProgressView()
                    } else {
                        Button("Kur", action: begin)
                            .disabled(requirement.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }

                Text(
                    "PyPI'den yalnızca saf Python (py3-none-any) tekerlekleri"
                        + " kurulabilir. numpy gibi derlenmiş uzantılar iOS'ta"
                        + " çalışmaz; böyle bir paket istenirse kurulum reddedilir."
                        + " Bağımlılıklar da aynı kuralla kurulur."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            if !store.transcript.isEmpty {
                Section("Son işlem") {
                    Text(store.transcript)
                        .font(.system(size: 12, design: .monospaced))
                        .textSelection(.enabled)
                }
            }

            Section("Kurulu paketler") {
                if store.installed.isEmpty {
                    Text("Henüz elle kurulmuş paket yok.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(store.installed) { package in
                        row(package)
                    }
                    .onDelete { offsets in
                        for index in offsets { store.remove(store.installed[index]) }
                    }
                }
                Text(
                    "Belgeler/Python/site-packages içinde dururlar; Dosyalar"
                        + " uygulamasından da görülebilirler ve uygulama"
                        + " güncellemesinde silinmezler."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Section("Uygulamayla gelen") {
                ForEach(store.bundled) { package in
                    row(package)
                }
                Text("Paketin içinde geldikleri için silinemezler.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Python paketleri")
        .navigationBarTitleDisplayMode(.inline)
        .scrollDismissesKeyboard(.interactively)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Bitti") { fieldFocused = false }
            }
        }
        .onAppear { store.refresh() }
    }

    private func row(_ package: Edge0PythonPackage) -> some View {
        HStack {
            Text(package.name)
            Spacer()
            Text(package.version)
                .foregroundStyle(.secondary)
                .font(.callout.monospacedDigit())
        }
    }

    private func begin() {
        let request = requirement
        fieldFocused = false
        Task {
            await store.install(request)
            requirement = ""
        }
    }
}
