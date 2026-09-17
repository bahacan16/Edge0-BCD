# Repo notes for Claude

## iOS builds

- Always produce **unsigned** IPAs from CI (`CODE_SIGNING_ALLOWED=NO`,
  `CODE_SIGNING_REQUIRED=NO`, `CODE_SIGN_IDENTITY=""`, `DEVELOPMENT_TEAM=""`).
- The user signs IPAs themselves afterwards using **Feather**, with their own
  `.p12` + `.mobileprovision` matching their device's UDID. Do not ask about
  code signing, provisioning profiles, Apple Developer accounts, or device
  UDIDs — this is already handled on the user's end for every project.
- Xcode-project SPM packages that ship Swift macros (e.g. `mlx-swift-lm`'s
  `MLXHuggingFaceMacros`) fail headless `xcodebuild` in CI with "Macro ...
  must be enabled before it can be used" unless `-skipMacroValidation` is
  passed to the `xcodebuild build` invocation.

## Script output files (decided 2026-09-17, not built yet)

Where the DXF→PIM script's `.pim` and `.png` go, and the rules around them.
Agreed with the user; build this alongside the Clipper2 / shapely-shim work.

- **Every path component is ASCII.** No Turkish characters anywhere in a
  folder or file name — these files end up on a USB stick and a Fagor
  controller. Transliterate what comes from the user's DXF name
  (`kapak plakası` → `kapak-plakasi`).
- **Folder names are English**, like the rest of the on-disk layout
  (`Models`, `Python`). So: `Documents/Outputs/`.
- **One folder per run, not per conversation.** A conversation can be
  returned to days later, so naming by chat start time puts new files under
  an old date; naming by chat and run time scatters one job across folders.
  The user looks for *the job for that part*, so the DXF's name identifies
  the folder and the timestamp separates repeats:

      Documents/Outputs/2026-09-17_0942_kapak-plakasi/
          kapak-plakasi.pim
          kapak-plakasi.png
          job.txt

- **Timestamps are `YYYY-MM-DD_HHmm`.** The Files app sorts names as text,
  and this is the only format that sorts chronologically. No spaces in the
  name: paths travel through shells and machine-side tooling.
- **`job.txt` is not optional.** It records the whole CONFIG block as run
  (MODE, POCKET_MODE, CUT_ORDER, tool diameter, sheet size), the source
  DXF's name and size, the model that ran it, the conversation title, the
  script version and the run time. When a part comes out wrong the question
  is "what settings cut this", and a re-cut costs more than a text file.
- **The app sets the output path, never the model.** Same rule as `MODE`:
  the model fills in the fields of CONFIG it is given and nothing else.
  Otherwise it can write anywhere, including over the script itself.
- **Do not exclude `Outputs` from iCloud backup.** `Models` is excluded
  because a 23 GB checkpoint has no business in a backup; outputs are a few
  hundred kilobytes and losing them is a real cost.
- **Offer a share sheet when a run finishes** ("PIM'i paylaş" / "Dosyalar'da
  göster"). AirDropping the `.pim` straight to the PC is the actual last
  step of the workflow; hunting for it in Files is not.
