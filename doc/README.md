# Documentación LaTeX

Este directorio contiene la documentación técnica completa del Predictor Estocástico Universal en formato LaTeX.

## 📁 Estructura de Directorios

```bash
doc/
├── *.tex                           # Archivos fuente LaTeX
├── pdf/                            # PDFs compilados (versionados)
├── build/                          # Artefactos de compilación (ignorados por git)
├── .latexmkrc                      # Configuración de compilación
└── compile.sh                      # Script de compilación
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

- **Directorio de artefactos**: `build/` (ignorado por git)
- **Directorio de salida**: `pdf/` (PDFs finales, versionados)
- **Compilador**: `pdflatex` con `synctex` habilitado
- **Limpieza automática**: Archivos auxiliares nunca ensucian el workspace

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

1. **Editar** archivos `.tex` en el directorio raíz
2. **Compilar** con `./compile.sh` o `./compile.sh <archivo>`
3. **Revisar** PDFs generados en `pdf/`
4. **Commit** solo archivos `.tex` y PDFs finales (no artefactos)

Los artefactos de compilación (`.aux`, `.log`, `.toc`, etc.) se generan automáticamente en `build/` y son ignorados por git.

## ✨ Ventajas de Esta Configuración

- ✅ **Workspace limpio**: Solo archivos fuente visibles
- ✅ **Compilación rápida**: `latexmk` gestiona dependencias automáticamente
- ✅ **Git amigable**: Artefactos no contaminan el historial
- ✅ **PDFs organizados**: Salida centralizada en `pdf/`
- ✅ **Reproducible**: Configuración versionada en `.latexmkrc`

## 🔧 Configuración del Editor

### VS Code (LaTeX Workshop)

Agregar a `.vscode/settings.json`:

```json
{
  "latex-workshop.latex.outDir": "pdf",
  "latex-workshop.latex.auxDir": "build"
}
```

### Overleaf / TeXstudio

Configurar directorio de salida en preferencias del proyecto.

## 📚 Referencias

- [latexmk documentation](https://mg.readthedocs.io/latexmk.html)
- [LaTeX project](https://www.latex-project.org/)
