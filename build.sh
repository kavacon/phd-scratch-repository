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

# Compile from the source directory so all relative paths in .tex files
# resolve correctly. Auxiliary files land alongside main.tex, then the
# final PDF is copied into build/.
OLD_DIR="$PWD"
cd "$SOURCE_DIR"

echo "--- pdflatex (1/4) ---"
pdflatex -interaction=nonstopmode main.tex > /tmp/build-latex.log 2>&1

echo "--- $BIB_BACKEND ---"
if [ "$BIB_BACKEND" = "biber" ]; then
    biber main
else
    bibtex -terse main
fi

echo "--- pdflatex (2/4) ---"
pdflatex -interaction=nonstopmode main.tex > /tmp/build-latex.log 2>&1

echo "--- pdflatex (3/4) ---"
pdflatex -interaction=nonstopmode main.tex > /tmp/build-latex.log 2>&1

# Move final PDF to build/ and clean auxiliary files from source dir
mv -f main.pdf "$BUILD_DIR/"
find "$SOURCE_DIR" -maxdepth 1 -type f \( -name "*.aux" -o -name "*.bcf" -o -name "*.bbl" -o -name "*.blg" -o -name "*.run.xml" -o -name "*.out" -o -name "*.toc" -o -name "*.log" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" \) -delete

cd "$OLD_DIR"

echo ""
echo "=== Done: $BUILD_DIR/main.pdf ==="
