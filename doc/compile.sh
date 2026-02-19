#!/bin/bash
# LaTeX Compiler - Universal Stochastic Predictor
# Structure:
#   latex/specification/ → .tex source files
#   latex/implementation/ → implementation docs
#   pdf/specification/ → compiled PDFs
#   pdf/implementation/ → compiled PDFs
#   .build/ → temporary artifacts (git-ignored)

set -e

# Base directory
DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOC_DIR"

# Ensure directories
mkdir -p "$DOC_DIR/.build" "$DOC_DIR/pdf/specification"

# Remove stale PDFs (no corresponding .tex)
cleanup_old_pdfs() {
    local source_dir="$1"
    local pdf_dir="$2"
    
    if [ ! -d "$pdf_dir" ]; then
        return
    fi
    
    for pdf_file in "$pdf_dir"/*.pdf; do
        if [ -f "$pdf_file" ]; then
            base_name=$(basename "$pdf_file" .pdf)
            if [ ! -f "$source_dir/$base_name.tex" ]; then
                rm -f "$pdf_file"
                echo "🗑️  Deleted: $(basename $pdf_file) (no matching .tex)"
            fi
        fi
    done
}

# Check whether a .tex file needs recompilation
needs_recompile() {
    local tex_file="$1"
    local pdf_dir="$2"
    local base_name=$(basename "$tex_file" .tex)
    local pdf_file="$pdf_dir/$base_name.pdf"
    
    if [ ! -f "$pdf_file" ]; then
        return 0  # true: PDF missing
    fi
    
    if [ "$tex_file" -nt "$pdf_file" ]; then
        return 0  # true: .tex newer than PDF
    fi
    
    return 1  # false: PDF up to date
}

# Compile a .tex file
compile_doc() {
    local tex_file="$1"
    local pdf_dir="$2"
    local base_name=$(basename "$tex_file" .tex)
    local log_file="$DOC_DIR/.build/$base_name.log"
    
    echo "📄 Compiling $base_name.tex..."
    
    # First pass
    if ! lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > "$log_file" 2>&1; then
        echo "❌ Error on first pass"
        _show_errors "$log_file" "$base_name"
        return 1
    fi
    
    # Second pass (cross-references)
    if ! lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > "$log_file" 2>&1; then
        echo "❌ Error on second pass"
        _show_errors "$log_file" "$base_name"
        return 1
    fi
    
    # Copy PDF to destination
    if [ -f "$DOC_DIR/.build/$base_name.pdf" ]; then
        cp "$DOC_DIR/.build/$base_name.pdf" "$pdf_dir/$base_name.pdf"
        echo "✅ $base_name.pdf → pdf/$(basename $pdf_dir)/"
    else
        echo "❌ $base_name.pdf was not generated"
        return 1
    fi
}

# Show LaTeX errors
_show_errors() {
    local log_file="$1"
    local base_name="$2"
    echo ""
    echo "🔴 ERRORS IN $base_name.tex:"
    grep -E "^.*\.tex:[0-9]+:" "$log_file" | head -10 || true
    grep -E "^!|^l\.[0-9]+" "$log_file" | head -10 || true
    echo "📋 Log: $log_file"
    echo ""
}

# Clean artifacts
clean_all() {
    echo "🧹 Cleaning artifacts..."
    rm -rf "$DOC_DIR/.build"
    mkdir -p "$DOC_DIR/.build"
    echo "✅ Cleanup complete"
}

# Parse arguments
case "${1:-help}" in
    help|-h|--help)
        echo "LaTeX Compiler - Stochastic Predictor"
        echo ""
        echo "Usage:"
        echo "  ./compile.sh                      # Show this help (default)"
        echo "  ./compile.sh <file>               # Compile specific file"
        echo "  ./compile.sh <file>.tex           # Compile specific file (with extension)"
        echo "  ./compile.sh --all                # Compile documents with changes"
        echo "  ./compile.sh --all --force        # Force compilation of all documents"
        echo "  ./compile.sh -a -f                # Short version of --all --force"
        echo "  ./compile.sh clean                # Clean all artifacts"
        echo ""
        echo "Examples:"
        echo "  ./compile.sh Stochastic_Predictor_Python       # Compile only Python.tex"
        echo "  ./compile.sh --all                             # Compile only changes"
        echo "  ./compile.sh --all --force                     # Recompile everything"
        ;;
    clean)
        clean_all
        ;;
    --all|-a|all)
        # Compile only changed files unless --force is specified
        force_recompile=false
        if [ "${2:-}" = "--force" ] || [ "${2:-}" = "-f" ]; then
            force_recompile=true
        fi
        
        echo "🚀 Compiling documents with changes..."
        if [ "$force_recompile" = true ]; then
            echo "   (--force mode: compile everything regardless of changes)"
            # Clean only temporary artifacts
            rm -rf "$DOC_DIR/.build"
            mkdir -p "$DOC_DIR/.build"
        fi
        echo ""
        
        compiled_count=0
        skipped_count=0
        
        # Process all folders under /doc/latex
        latex_dir="$DOC_DIR/latex"
        if [ ! -d "$latex_dir" ]; then
            echo "❌ Directory does not exist: $latex_dir"
            exit 1
        fi
        
        # Iterate over each folder in latex/
        for source_dir in "$latex_dir"/*/; do
            # Obtener nombre de la carpeta
            phase=$(basename "$source_dir")
            pdf_dir="$DOC_DIR/pdf/$phase"
            
            # Crear directorio PDF correspondiente si no existe
            mkdir -p "$pdf_dir"
            
            echo "📦 Processing folder: $phase"
            
            # Limpiar PDFs obsoletos
            echo "🧹 Removing stale PDFs in $phase..."
            cleanup_old_pdfs "$source_dir" "$pdf_dir"
            
            # Compilar todos los .tex de esta carpeta
            for tex_file in "$source_dir"/*.tex; do
                if [ -f "$tex_file" ]; then
                    base_name=$(basename "$tex_file" .tex)
                    
                    # Check if compilation is needed
                    if [ "$force_recompile" = true ] || needs_recompile "$tex_file" "$pdf_dir"; then
                        if compile_doc "$tex_file" "$pdf_dir"; then
                            ((compiled_count++))
                        else
                            echo "⚠️  Compilation failed for $base_name.tex"
                        fi
                    else
                        echo "⏭️  $base_name.tex unchanged, skipping..."
                        ((skipped_count++))
                    fi
                fi
            done
            echo ""
        done
        
        echo "📊 Summary: $compiled_count compiled, $skipped_count skipped"
        if [ $compiled_count -gt 0 ]; then
            echo "✨ Compilation complete. PDFs generated in: $DOC_DIR/pdf/"
        else
            echo "ℹ️  All documents are up to date."
        fi
        ;;
    *)
        # Compile specific file
        tex_file=""
        filename="${1%.tex}"  # Strip .tex if provided
        
        # Look for file in latex/specification
        source_dir="$DOC_DIR/latex/specification"
        pdf_dir="$DOC_DIR/pdf/specification"
        
        if [ -f "$source_dir/$filename.tex" ]; then
            tex_file="$source_dir/$filename.tex"
        elif [ -f "$filename.tex" ]; then
            tex_file="$filename.tex"
        elif [ -f "$filename" ]; then
            tex_file="$filename"
        else
            echo "❌ File not found: ${1}"
            echo ""
            echo "Usage: ./compile.sh <file> | --all | --all --force | clean | help"
            exit 1
        fi
        
        compile_doc "$tex_file" "$pdf_dir"
        ;;
esac

