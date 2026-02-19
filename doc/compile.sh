#!/bin/bash
# Compilador LaTeX - Universal Stochastic Predictor
# Estructura:
#   latex/specification/ → .tex source files
#   latex/implementation/ → future implementation docs (TBD)
#   pdf/specification/ → compiled PDFs
#   pdf/implementation/ → future implementation PDFs (TBD)
#   .build/ → temporary artifacts (git-ignored)

set -e

# Directorio base
DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOC_DIR"

# Asegurar directorios
mkdir -p "$DOC_DIR/.build" "$DOC_DIR/pdf/specification"

# Limpiar PDFs obsoletos (sin .tex correspondiente)
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
                echo "🗑️  Borrado: $(basename $pdf_file) (sin .tex correspondiente)"
            fi
        fi
    done
}

# Verificar si archivo .tex necesita recompilación
needs_recompile() {
    local tex_file="$1"
    local pdf_dir="$2"
    local base_name=$(basename "$tex_file" .tex)
    local pdf_file="$pdf_dir/$base_name.pdf"
    
    if [ ! -f "$pdf_file" ]; then
        return 0  # true: no existe PDF
    fi
    
    if [ "$tex_file" -nt "$pdf_file" ]; then
        return 0  # true: .tex más nuevo que PDF
    fi
    
    return 1  # false: PDF actualizado
}

# Compilar un archivo .tex
compile_doc() {
    local tex_file="$1"
    local pdf_dir="$2"
    local base_name=$(basename "$tex_file" .tex)
    local log_file="$DOC_DIR/.build/$base_name.log"
    
    echo "📄 Compilando $base_name.tex..."
    
    # Primera pasada
    if ! lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > "$log_file" 2>&1; then
        echo "❌ Error en primera pasada"
        _show_errors "$log_file" "$base_name"
        return 1
    fi
    
    # Segunda pasada (referencias cruzadas)
    if ! lualatex -interaction=nonstopmode \
             -file-line-error \
             -synctex=1 \
             -output-directory="$DOC_DIR/.build" \
             "$tex_file" > "$log_file" 2>&1; then
        echo "❌ Error en segunda pasada"
        _show_errors "$log_file" "$base_name"
        return 1
    fi
    
    # Copiar PDF al destino
    if [ -f "$DOC_DIR/.build/$base_name.pdf" ]; then
        cp "$DOC_DIR/.build/$base_name.pdf" "$pdf_dir/$base_name.pdf"
        echo "✅ $base_name.pdf → pdf/$(basename $pdf_dir)/"
    else
        echo "❌ No se generó $base_name.pdf"
        return 1
    fi
}

# Mostrar errores LaTeX
_show_errors() {
    local log_file="$1"
    local base_name="$2"
    echo ""
    echo "🔴 ERRORES EN $base_name.tex:"
    grep -E "^.*\.tex:[0-9]+:" "$log_file" | head -10 || true
    grep -E "^!|^l\.[0-9]+" "$log_file" | head -10 || true
    echo "📋 Log: $log_file"
    echo ""
}

# Limpiar artefactos
clean_all() {
    echo "🧹 Limpiando artefactos..."
    rm -rf "$DOC_DIR/.build"
    mkdir -p "$DOC_DIR/.build"
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
        
        # Procesar TODAS las carpetas en /doc/latex
        latex_dir="$DOC_DIR/latex"
        if [ ! -d "$latex_dir" ]; then
            echo "❌ No existe directorio: $latex_dir"
            exit 1
        fi
        
        # Iterar sobre cada carpeta en latex/
        for source_dir in "$latex_dir"/*/; do
            # Obtener nombre de la carpeta
            phase=$(basename "$source_dir")
            pdf_dir="$DOC_DIR/pdf/$phase"
            
            # Crear directorio PDF correspondiente si no existe
            mkdir -p "$pdf_dir"
            
            echo "📦 Procesando carpeta: $phase"
            
            # Limpiar PDFs obsoletos
            echo "🧹 Limpiando PDFs obsoletos en $phase..."
            cleanup_old_pdfs "$source_dir" "$pdf_dir"
            
            # Compilar todos los .tex de esta carpeta
            for tex_file in "$source_dir"/*.tex; do
                if [ -f "$tex_file" ]; then
                    base_name=$(basename "$tex_file" .tex)
                    
                    # Verificar si necesita compilación
                    if [ "$force_recompile" = true ] || needs_recompile "$tex_file" "$pdf_dir"; then
                        if compile_doc "$tex_file" "$pdf_dir"; then
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
        done
        
        echo "📊 Resumen: $compiled_count compilados, $skipped_count omitidos"
        if [ $compiled_count -gt 0 ]; then
            echo "✨ Compilación completa. PDFs generados en: $DOC_DIR/pdf/"
        else
            echo "ℹ️  Todos los documentos están actualizados."
        fi
        ;;
    *)
        # Compilar archivo específico
        tex_file=""
        filename="${1%.tex}"  # Remover .tex si está puesto
        
        # Buscar archivo en latex/specification
        source_dir="$DOC_DIR/latex/specification"
        pdf_dir="$DOC_DIR/pdf/specification"
        
        if [ -f "$source_dir/$filename.tex" ]; then
            tex_file="$source_dir/$filename.tex"
        elif [ -f "$filename.tex" ]; then
            tex_file="$filename.tex"
        elif [ -f "$filename" ]; then
            tex_file="$filename"
        else
            echo "❌ Archivo no encontrado: ${1}"
            echo ""
            echo "Uso: ./compile.sh <archivo> | --all | --all --force | clean | help"
            exit 1
        fi
        
        compile_doc "$tex_file" "$pdf_dir"
        ;;
esac

