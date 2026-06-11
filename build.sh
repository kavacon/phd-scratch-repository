#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage: $(basename "$0") <target>"
    echo ""
    echo "Targets:"
    echo "  thesis"
    for dir in "$REPO_ROOT"/publications/publication_*; do
        [ -d "$dir" ] && echo "  $(basename "$dir")"
    done
    for dir in "$REPO_ROOT"/reports/report_*; do
        [ -d "$dir" ] && echo "  $(basename "$dir")"
    done
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

TARGET="$1"

case "$TARGET" in
    thesis)
        SOURCE_DIR="$REPO_ROOT/thesis"
        ;;
    publication_*)
        SOURCE_DIR="$REPO_ROOT/publications/$TARGET"
        ;;
    report_*)
        SOURCE_DIR="$REPO_ROOT/reports/$TARGET"
        ;;
    *)
        echo "Error: Unknown target '$TARGET'"
        usage
        ;;
esac

MAIN_TEX="$SOURCE_DIR/main.tex"
BUILD_DIR="$SOURCE_DIR/build"

if [ ! -f "$MAIN_TEX" ]; then
    echo "Error: '$MAIN_TEX' not found"
    exit 1
fi

mkdir -p "$BUILD_DIR"

# Detect bibliography backend
if grep -qE '\\addbibresource|\\printbibliography|preamble/bibliography' "$MAIN_TEX"; then
    BIB_BACKEND="biber"
else
    BIB_BACKEND="bibtex"
fi

echo "=== Compiling $TARGET (backend: $BIB_BACKEND) ==="
echo "Source:  $SOURCE_DIR"
echo "Build:   $BUILD_DIR"
echo ""

cd "$SOURCE_DIR"

PDFLATEX="pdflatex -interaction=nonstopmode -output-directory=$BUILD_DIR"

echo "--- pdflatex (1/3) ---"
$PDFLATEX main.tex

echo "--- $BIB_BACKEND ---"
if [ "$BIB_BACKEND" = "biber" ]; then
    BIBINPUTS="$REPO_ROOT" biber --input-directory="$BUILD_DIR" --output-directory="$BUILD_DIR" "$BUILD_DIR/main"
    if [ $? -ne 0 ]; then
        echo "ERROR: Biber failed"
        exit 1
    fi
else
    bibtex -terse "$BUILD_DIR/main"
fi

echo "--- pdflatex (2/3) ---"
$PDFLATEX main.tex

echo "--- pdflatex (3/3) ---"
$PDFLATEX main.tex

echo ""
echo "=== Done: $BUILD_DIR/main.pdf ==="