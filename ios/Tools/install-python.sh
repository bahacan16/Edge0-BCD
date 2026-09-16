#!/bin/bash
# Xcode build phase: lay the Python runtime out inside the app bundle.
#
# Most of this is BeeWare's own `install_python`, sourced rather than
# reimplemented — it copies the stdlib into the bundle and turns each of the
# two hundred `.so` extension modules into `Frameworks/<module>.framework/`,
# leaving a `.fwork` placeholder where the `.so` was for CPython's iOS loader
# to follow. Getting that wrong is the kind of thing only a device reveals, so
# it is not worth rewriting.
#
# One function is replaced: theirs finishes by running codesign with
# `$EXPANDED_CODE_SIGN_IDENTITY`, and this app is built unsigned on purpose —
# the IPA is signed afterwards with the user's own certificate. With signing
# off that variable is empty and `set -e` would fail the build on a step we do
# not want anyway. The frameworks land in `Frameworks/`, which is exactly where
# every signer looks, so they get signed with everything else later.
set -euo pipefail

XCFRAMEWORK="${1:-Python.xcframework}"
shift || true

# Normally set by Xcode; with signing disabled it is worth not depending on.
export CODESIGNING_FOLDER_PATH="${CODESIGNING_FOLDER_PATH:-$BUILT_PRODUCTS_DIR/$CONTENTS_FOLDER_PATH}"
export PLATFORM_FAMILY_NAME="${PLATFORM_FAMILY_NAME:-iOS}"

source "$PROJECT_DIR/$XCFRAMEWORK/build/utils.sh"

# Transcribed from the sourced file, minus the signing. Defined after the
# source so it wins: `process_dylibs` resolves it when it calls, not when it
# was defined.
install_dylib () {
    PYTHON_XCFRAMEWORK_PATH=$1
    INSTALL_BASE=$2
    FULL_EXT=$3

    EXT=$(basename "$FULL_EXT")
    MODULE_PATH=$(dirname "$FULL_EXT")
    MODULE_NAME=$(echo "$EXT" | cut -d "." -f 1)
    RELATIVE_EXT=${FULL_EXT#$CODESIGNING_FOLDER_PATH/}
    PYTHON_EXT=${RELATIVE_EXT/$INSTALL_BASE/}
    FULL_MODULE_NAME=$(echo "$PYTHON_EXT" | cut -d "." -f 1 | tr "/" ".")
    FRAMEWORK_BUNDLE_ID=$(echo "$PRODUCT_BUNDLE_IDENTIFIER.$FULL_MODULE_NAME" | tr "_" "-")
    FRAMEWORK_FOLDER="Frameworks/$FULL_MODULE_NAME.framework"

    if [ ! -d "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER" ]; then
        mkdir -p "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER"
        cp "$PROJECT_DIR/$PYTHON_XCFRAMEWORK_PATH/build/$PLATFORM_FAMILY_NAME-dylib-Info-template.plist" \
            "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER/Info.plist"
        plutil -replace CFBundleExecutable -string "$FULL_MODULE_NAME" \
            "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER/Info.plist"
        plutil -replace CFBundleIdentifier -string "$FRAMEWORK_BUNDLE_ID" \
            "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER/Info.plist"
    fi

    mv "$FULL_EXT" "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER/$FULL_MODULE_NAME"
    # The placeholder CPython's iOS loader follows to find the real binary.
    echo "$FRAMEWORK_FOLDER/$FULL_MODULE_NAME" > "${FULL_EXT%.so}.fwork"
    echo "${RELATIVE_EXT%.so}.fwork" \
        > "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER/$FULL_MODULE_NAME.origin"

    if [ -e "$MODULE_PATH/$MODULE_NAME.xcprivacy" ]; then
        rm -rf "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER/PrivacyInfo.xcprivacy"
        mv "$MODULE_PATH/$MODULE_NAME.xcprivacy" \
            "$CODESIGNING_FOLDER_PATH/$FRAMEWORK_FOLDER/PrivacyInfo.xcprivacy"
    fi
}

install_python "$XCFRAMEWORK" "$@"

# Proof that the packaging did what it claims. The device answers whether the
# interpreter runs; this answers whether it was assembled, and the two failures
# look nothing alike once they are separated.
#
# Written to a file as well as the log. The build step's own output runs to
# tens of thousands of lines and job logs can only be read from the end, so a
# report printed here is a report that cannot be retrieved — which has already
# cost this work two rounds.
REPORT="$PROJECT_DIR/python-install-report.txt"
exec > >(tee "$REPORT") 2>&1

echo "== bundle python layout =="
ls -1 "$CODESIGNING_FOLDER_PATH/python/lib" 2>/dev/null || echo "NO python/lib"
echo "-- framework count: $(find "$CODESIGNING_FOLDER_PATH/Frameworks" -maxdepth 1 -name '*.framework' 2>/dev/null | wc -l | tr -d ' ')"
echo "-- .fwork count:    $(find "$CODESIGNING_FOLDER_PATH/python" -name '*.fwork' 2>/dev/null | wc -l | tr -d ' ')"
echo "-- leftover .so:    $(find "$CODESIGNING_FOLDER_PATH/python" "$CODESIGNING_FOLDER_PATH/app_packages" -name '*.so' 2>/dev/null | wc -l | tr -d ' ')"
for want in zlib binascii _struct math; do
  if [ -d "$CODESIGNING_FOLDER_PATH/Frameworks/$want.framework" ]; then
    echo "-- $want.framework ok"
  else
    echo "-- $want.framework MISSING"
  fi
done
echo "-- app_packages: $(ls -1 "$CODESIGNING_FOLDER_PATH/app_packages" 2>/dev/null | tr '\n' ' ')"
