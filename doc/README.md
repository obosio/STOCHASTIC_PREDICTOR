# Documentación LaTeX

Este directorio contiene la documentación técnica completa del Predictor Estocástico Universal en formato LaTeX.

## 📁 Estructura de Directorios

```bash
doc/
├── *.tex                           # Archivos fuente LaTeX (7 documentos)
├── pdf/                            # PDFs compilados (versionados en git)
├── .build/                         # Artefactos de compilación (oculto, ignorado por git)
├── .latexmkrc                      # Configuración de compilación (lualatex)
├── compile.sh                      # Script de compilación bash
└── README.md                       # Este archivo
```

## 📄 Documentos Disponibles

### Documentos Teóricos y Generales

- **Predictor_Estocastico_Teoria.tex** - Fundamentos matemáticos y teoremas
- **Predictor_Estocastico_Implementacion.tex** - Algoritmos y métodos numéricos
- **Predictor_Estocastico_Pruebas.tex** - Protocolo de validación y pruebas (agnóstico de lenguaje)
- **Predictor_Estocastico_IO.tex** - Especificación de I/O y telemetría

### Documentos Específicos de Python

- **Predictor_Estocastico_Python.tex** - Guía de implementación en Python con JAX
- **Predictor_Estocastico_API_Python.tex** - Especificación de API Python
- **Predictor_Estocastico_Tests_Python.tex** - Suite de pruebas en Python/pytest

## 🚀 Compilación

### Compilar todos los documentos

```bash
./compile.sh
```

### Compilar un documento específico

```bash
./compile.sh Predictor_Estocastico_Teoria.tex
# O simplemente:
./compile.sh Predictor_Estocastico_Teoria
```

### Limpiar artefactos de compilación

```bash
./compile.sh clean
```

## 🎯 Configuración Automática

El archivo `.latexmkrc` configura automáticamente:

- **Directorio de artefactos**: `.build/` (oculto, ignorado por git)
- **Directorio de salida**: `pdf/` (PDFs finales, versionados)
- **Compilador**: `lualatex` (LuaTeX/XeTeX) con `synctex` habilitado
- **Limpieza automática**: Archivos auxiliares (`.aux`, `.log`, `.toc`, etc.) generados en `.build/`
- **Integración git**: `.build/` excluido por `.gitignore`, solo `.tex` y `pdf/` versionados

## 🛠️ Requisitos

### LaTeX

```bash
# macOS con MacTeX
brew install --cask mactex

# O instalación minimal
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install latexmk
```

### Paquetes LaTeX Necesarios

- `babel[spanish]`
- `fontspec`
- `amsmath`, `amssymb`, `amsthm`
- `listings`, `xcolor`
- `hyperref`
- `geometry`, `booktabs`

## 📝 Flujo de Trabajo

1. **Editar** archivos `.tex` en el directorio raíz (`doc/`)
2. **Compilar** con `./compile.sh all` o `./compile.sh <archivo>` (sin extensión `.tex`)
3. **Revisar** PDFs generados en `pdf/`
4. **Commit** solo archivos `.tex` y PDFs finales (no artefactos)

Los artefactos de compilación (`.aux`, `.log`, `.toc`, etc.) se generan automáticamente en `.build/` (oculto) y son ignorados por git. La limpieza se realiza con `./compile.sh clean`.

## 📊 Estado Actual (Febrero 2026)

**Documentos compilados exitosamente:**

- ✅ Predictor_Estocastico_Teoria.tex (228 KB)
- ✅ Predictor_Estocastico_Implementacion.tex (226 KB)
- ✅ Predictor_Estocastico_IO.tex (165 KB)
- ✅ Predictor_Estocastico_Pruebas.tex (256 KB)
- ✅ Predictor_Estocastico_Python.tex (32 páginas con mejoras de robustez)
- ✅ Predictor_Estocastico_API_Python.tex (10 páginas con hardening producción)
- ✅ Predictor_Estocastico_Tests_Python.tex (33 páginas con testing avanzado)

**Mejoras recientes:**

- Optimización de memoria en WTMM (compute_cwt_windowed)
- Gestión de precisión JAX (jax_enable_x64)
- Annealing de entropía en algoritmo JKO
- Versionado de schema en API
- Dump de emergencia para depuración
- Fuzzing con hypothesis
- Tests FPGA Q16.16
- Validación de causalidad

## ✨ Ventajas de Esta Configuración

- ✅ **Workspace limpio**: Solo archivos fuente visibles (artefactos en `.build/` oculto)
- ✅ **Compilación rápida**: `latexmk` gestiona dependencias y paralelización automáticamente
- ✅ **Git amigable**: Artefactos no contaminan el historial; solo PDFs finales versionados
- ✅ **PDFs organizados**: Salida centralizada en `pdf/`, históricamente preservada
- ✅ **Reproducible**: Configuración versionada en `.latexmkrc` y `compile.sh`
- ✅ **LuaTeX moderno**: Soporte nativo para Unicode, fuentes OpenType, características avanzadas

## 🔧 Configuración del Editor

### VS Code (LaTeX Workshop)

Agregar a `.vscode/settings.json`:

```json
{
  "latex-workshop.latex.outDir": "pdf",
  "latex-workshop.latex.auxDir": ".build",
  "files.exclude": {
    "**/.*": true
  }
}
```

El parámetro `files.exclude` oculta el directorio `.build/` en el explorador de archivos.

### Overleaf / TeXstudio

Configurar directorio de salida en preferencias del proyecto.

## ⚠️ Avisos de Compilación Conocidos

Se reportan advertencias menores sobre caracteres faltantes en fuentes monoespaciadas:

- Símbolos griegos (κ, γ, ρ) en `\texttt{}`/`\lstlisting`
- Caracteres especiales de caja de dibujo (├, ─, etc.)

**Impacto**: Cosmético. Los PDFs se generan completamente sin errores; las advertencias solo indican sustituciones de fuentes en entornos monoespaciados.

**Solución** (si es necesario): Usar fuentes específicas que soporten Unicode completo o reemplazar caracteres griegos con `\ensuremath{}`.

## 📚 Referencias

- [latexmk documentation](https://mg.readthedocs.io/latexmk.html)
- [LaTeX project](https://www.latex-project.org/)
