# Configuración de latexmk para mantener limpio el workspace
# Los archivos auxiliares se generarán en .build/

# Directorio para archivos auxiliares
$aux_dir = '.build';
$out_dir = 'pdf';

# Asegurar que los directorios existan
system("mkdir -p .build pdf");

# Usar xelatex (necesario para fontspec)
$pdf_mode = 5;  # 5 = xelatex
$postscript_mode = 0;
$dvi_mode = 0;

# Configuración de compilación
$xelatex = 'xelatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

# Limpieza completa
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fls nav snm vrb run tdo';

# Número máximo de iteraciones
$max_repeat = 5;
