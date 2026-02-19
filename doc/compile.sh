#!/bin/bash
# Script para compilar documentos LaTeX de forma limpia
# Todos los artefactos van a doc/.build/ y PDFs a doc/pdf/

set -e

# Directorio base
DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOC_DIR"

# Asegurar que existen los directorios (sin borrar PDFs previos)
mkdir -p "$DOC_DIR/.build" "$DOC_DIR/pdf"

# Función para compilar un archivo con lualatex directo (dos pasadas para actualizar referencias)
compile_doc() {
    local tex_file="$1"
    local base_name=$(basename "$tex_file" .tex)
    
    echo "📄 Compilando $base_name.tex con lualatex..."
    
    # Primera pasada: generar .aux con referencias
    lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > /dev/null 2>&1
    
    # Segunda pasada: resolver referencias cruzadas y tabla de contenidos
    lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > /dev/null 2>&1
    
    # Copiar PDF al directorio pdf/
    if [ -f "$DOC_DIR/.build/$base_name.pdf" ]; then
        cp "$DOC_DIR/.build/$base_name.pdf" "$DOC_DIR/pdf/$base_name.pdf"
        echo "✅ $base_name.pdf generado en pdf/"
    else
        echo "❌ Error: No se generó $base_name.pdf"
        return 1
    fi
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
