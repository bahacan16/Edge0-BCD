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
