#!/bin/bash
# Script para compilar documentos LaTeX de forma limpia
# Todos los artefactos van a doc/.build/ y PDFs a doc/pdf/

set -e

# Directorio base
DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOC_DIR"

# Asegurar que existen los directorios (sin borrar PDFs previos)
mkdir -p "$DOC_DIR/.build" "$DOC_DIR/pdf"

# Función para compilar un archivo
compile_doc() {
    local tex_file="$1"
    local base_name=$(basename "$tex_file" .tex)
    
    echo "📄 Compilando $base_name.tex..."
    
    latexmk -lualatex \
            -auxdir="$DOC_DIR/.build" \
            -outdir="$DOC_DIR/pdf" \
            -interaction=nonstopmode \
            -file-line-error \
            "$tex_file"
    
    echo "✅ $base_name.pdf generado en pdf/"
}

# Función para limpiar
clean_all() {
    echo "🧹 Limpiando artefactos de compilación..."
    latexmk -C -auxdir="$DOC_DIR/.build" -outdir="$DOC_DIR/pdf"
    rm -rf "$DOC_DIR/.build" "$DOC_DIR/pdf"
    mkdir -p "$DOC_DIR/.build" "$DOC_DIR/pdf"
    echo "✅ Limpieza completa"
}

# Parsear argumentos
case "${1:-all}" in
    clean)
        clean_all
        ;;
    all)
        echo "🚀 Compilando todos los documentos..."
        # Limpiar solo los artefactos temporales de compilaciones anteriores
        rm -rf "$DOC_DIR/.build"
        mkdir -p "$DOC_DIR/.build"
        for tex_file in *.tex; do
            if [ -f "$tex_file" ]; then
                compile_doc "$tex_file"
            fi
        done
        echo ""
        echo "✨ Compilación completa. PDFs en: $DOC_DIR/pdf/"
        ;;
    *)
        # Compilar archivo específico
        if [ -f "${1}" ]; then
            compile_doc "${1}"
        elif [ -f "${1}.tex" ]; then
            compile_doc "${1}.tex"
        else
            echo "❌ Archivo no encontrado: ${1}"
            echo ""
            echo "Uso:"
            echo "  ./compile.sh              # Compila todos los .tex"
            echo "  ./compile.sh <archivo>    # Compila archivo específico"
            echo "  ./compile.sh clean        # Limpia artefactos"
            exit 1
        fi
        ;;
esac
