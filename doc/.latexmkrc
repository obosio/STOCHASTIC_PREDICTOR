# Configuración de latexmk para mantener limpio el workspace
# Los archivos auxiliares se generarán en .build/

# Directorio para archivos auxiliares
$aux_dir = '.build';
$out_dir = '.build';  # También salida a .build para simplicidad

# Asegurar que los directorios existan
system("mkdir -p .build");

# FORZAR lualatex como motor de PDF
$pdf_mode = 4;  # 4 = lualatex, no pdflatex
$postscript_mode = 0;
$dvi_mode = 0;

# Definir el comando de lualatex explícitamente
$lualatex = 'lualatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

# Limpieza completa
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fls nav snm vrb run tdo';

# Número máximo de iteraciones
$max_repeat = 5;

# Asegurar que no se intente pdflatex bajo ningún caso
# Forceamos la evaluación de pdf_mode después de cualquier otro setting
