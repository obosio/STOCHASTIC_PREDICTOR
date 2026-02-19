#!/bin/bash
# Script para compilar documentos LaTeX de forma limpia
# Todos los artefactos van a doc/.build/ y PDFs a doc/pdf/

set -e

# Directorio base
DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOC_DIR"

# Asegurar que existen los directorios (sin borrar PDFs previos)
mkdir -p "$DOC_DIR/.build" "$DOC_DIR/pdf"

# Función que verifica si un archivo .tex ha cambiado respecto a su PDF
needs_recompile() {
    local tex_file="$1"
    local base_name=$(basename "$tex_file" .tex)
    local pdf_file="$DOC_DIR/pdf/$base_name.pdf"
    
    # Si el PDF no existe, necesita compilación
    if [ ! -f "$pdf_file" ]; then
        return 0  # true: necesita compilación
    fi
    
    # Si el .tex es más nuevo que el PDF, necesita compilación
    if [ "$tex_file" -nt "$pdf_file" ]; then
        return 0  # true: necesita compilación
    fi
    
    return 1  # false: no necesita compilación
}

# Función para compilar un archivo con lualatex directo (dos pasadas para actualizar referencias)
compile_doc() {
    local tex_file="$1"
    local base_name=$(basename "$tex_file" .tex)
    local log_file="$DOC_DIR/.build/$base_name.log"
    
    echo "📄 Compilando $base_name.tex con lualatex..."
    
    # Primera pasada: generar .aux con referencias
    if ! lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > "$log_file" 2>&1; then
        echo "❌ Error en primera pasada de compilación"
        # Mostrar errores relevantes del log
        _show_latex_errors "$log_file" "$base_name"
        return 1
    fi
    
    # Segunda pasada: resolver referencias cruzadas y tabla de contenidos
    if ! lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > "$log_file" 2>&1; then
        echo "❌ Error en segunda pasada de compilación"
        _show_latex_errors "$log_file" "$base_name"
        return 1
    fi
    
    # Verificar si se generó el PDF
    if [ -f "$DOC_DIR/.build/$base_name.pdf" ]; then
        cp "$DOC_DIR/.build/$base_name.pdf" "$DOC_DIR/pdf/$base_name.pdf"
        echo "✅ $base_name.pdf generado en pdf/"
    else
        echo "❌ Error: No se generó $base_name.pdf"
        echo "📋 Log guardado en: $log_file"
        return 1
    fi
}

# Función para mostrar errores de LaTeX de forma legible
_show_latex_errors() {
    local log_file="$1"
    local base_name="$2"
    
    echo ""
    echo "🔴 ERRORES ENCONTRADOS EN $base_name.tex:"
    echo "─────────────────────────────────────────"
    
    # Extraer líneas con errores (formato: archivo:línea:error)
    grep -E "^.*\.tex:[0-9]+:" "$log_file" | head -20 || true
    
    # Extraer líneas con "!" (errores LaTeX)
    grep -E "^!|^l\.[0-9]+" "$log_file" | head -20 || true
    
    echo "─────────────────────────────────────────"
    echo "📋 Log completo disponible en:"
    echo "   $log_file"
    echo ""
}

# Función para limpiar
clean_all() {
    echo "🧹 Limpiando artefactos de compilación..."
    latexmk -C -auxdir="$DOC_DIR/.build" -outdir="$DOC_DIR/pdf" 2>/dev/null || true
    rm -rf "$DOC_DIR/.build" "$DOC_DIR/pdf"
    mkdir -p "$DOC_DIR/.build" "$DOC_DIR/pdf"
    echo "✅ Limpieza completa"
}

# Parsear argumentos
case "${1:-help}" in
    help|-h|--help)
        echo "Compilador de LaTeX - Stochastic Predictor"
        echo ""
        echo "Uso:"
        echo "  ./compile.sh                      # Muestra esta ayuda (por defecto)"
        echo "  ./compile.sh <archivo>            # Compila archivo específico"
        echo "  ./compile.sh <archivo>.tex        # Compila archivo específico (con extensión)"
        echo "  ./compile.sh --all                # Compila documentos con cambios"
        echo "  ./compile.sh --all --force        # Fuerza compilación de todos los documentos"
        echo "  ./compile.sh -a -f                # Versión corta de --all --force"
        echo "  ./compile.sh clean                # Limpia todos los artefactos"
        echo ""
        echo "Ejemplos:"
        echo "  ./compile.sh Predictor_Estocastico_Python      # Compila solo Python.tex"
        echo "  ./compile.sh --all                             # Compila solo cambios"
        echo "  ./compile.sh --all --force                     # Recompila todo"
        ;;
    clean)
        clean_all
        ;;
    --all|-a|all)
        # Compilar solo archivos que han cambiado, a menos que se especifique --force
        force_recompile=false
        if [ "${2:-}" = "--force" ] || [ "${2:-}" = "-f" ]; then
            force_recompile=true
        fi
        
        echo "🚀 Compilando documentos con cambios..."
        if [ "$force_recompile" = true ]; then
            echo "   (modo --force: compilará todos sin importar cambios)"
            # Limpiar solo los artefactos temporales
            rm -rf "$DOC_DIR/.build"
            mkdir -p "$DOC_DIR/.build"
        fi
        echo ""
        
        compiled_count=0
        skipped_count=0
        
        for tex_file in Predictor_Estocastico_*.tex; do
            if [ -f "$tex_file" ]; then
                base_name=$(basename "$tex_file" .tex)
                
                # Verificar si necesita compilación
                if [ "$force_recompile" = true ] || needs_recompile "$tex_file"; then
                    if compile_doc "$tex_file"; then
                        ((compiled_count++))
                    else
                        echo "⚠️  Falló compilación de $base_name.tex"
                    fi
                else
                    echo "⏭️  $base_name.tex sin cambios, omitiendo..."
                    ((skipped_count++))
                fi
            fi
        done
        
        echo ""
        echo "📊 Resumen: $compiled_count compilados, $skipped_count omitidos"
        if [ $compiled_count -gt 0 ]; then
            echo "✨ Compilación completa. PDFs en: $DOC_DIR/pdf/"
        else
            echo "ℹ️  Todos los documentos están actualizados."
        fi
        ;;
    *)
        # Compilar archivo específico
        tex_file=""
        if [ -f "${1}" ]; then
            tex_file="${1}"
        elif [ -f "${1}.tex" ]; then
            tex_file="${1}.tex"
        else
            echo "❌ Archivo no encontrado: ${1}"
            echo ""
            echo "Uso: ./compile.sh <archivo> | --all | --all --force | clean | help"
            exit 1
        fi
        
        compile_doc "$tex_file"
        ;;
esac

