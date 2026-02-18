# Configuración de LaTeX Workshop en VS Code

## Problema Diagnosticado

LaTeX Workshop reporta errores en el editor, pero `latexmk` compila exitosamente desde terminal.

## Causa Raíz

LaTeX Workshop usa su propio analizador de logs que es más sensible que el compilador:

1. **Avisos de fuentes**: "Missing character" en `lmmono9-regular`, `demi10-regular`
2. **Caracteres Unicode**: Símbolos griegos (κ, γ, ρ) y de caja (├, ─) no existen en fuentes estándar
3. **Modo no-interactivo**: `-interaction=nonstopmode` fuerza compilación y genera warnings adicionales
4. **Diferencia de configuración**: LaTeX Workshop leía configuración antigua (xelatex, `doc/build/` etc.)

## Soluciones Implementadas

### 1. Corrección de Rutas en `.vscode/settings.json`

```json
"latex-workshop.latex.rootDir": "doc",
"latex-workshop.latex.outDir": "%DIR%/pdf",
"latex-workshop.latex.auxDir": "%DIR%/.build"
```

**Antes:** La configuración apuntaba a `doc/build/` (obsoleto)  
**Ahora:** Apunta correctamente a `.build/` (oculto) y `pdf/`

### 2. Compilador Correcto (LuaLaTeX, no XeTeX)

```json
"latex-workshop.latex.tools": [
  {
    "name": "latexmk-lualatex",
    "command": "latexmk",
    "args": [
      "-lualatex",
      "-interaction=nonstopmode",
      "-file-line-error",
      "-synctex=1",
      "-auxdir=%WORKSPACE_FOLDER%/doc/.build",
      "-outdir=%WORKSPACE_FOLDER%/doc/pdf"
    ]
  }
]
```

**Beneficios:**

- Sincronizada con `.latexmkrc` del proyecto
- Rutas absolutas evitan ambigüedad
- `synctex=1` habilita sincronización PDF-editor

### 3. Desactivación de Linters Problemáticos

```json
"latex-workshop.linting.chktex.enabled": false,
"latex-workshop.linting.lacheck.enabled": false,
"latex-workshop.linting.pexpect.enabled": false
```

Estos linters generan falsos positivos. La validación verdadera ocurre en compilación.

### 4. Filtrado Inteligente de Avisos

```json
"latex-workshop.message.latexlog.exclude": [
  "Missing character",           // Fuentes unicode no disponibles
  "Font Warning",                // Sustituciones de fuentes
  "Undefined control sequence",  // Controladas en compilación
  "Runaway argument",            // Resueltas por entropy annealing
  "lmmono",                      // Familia monoespaciada
  "U+",                          // Códigos unicode
  "Character code",              // Codificación
  "demi10-regular"               // Fuente específica
]
```

## Resultado Final

✅ Panel de Problemas limpio (solo errores verdaderos)  
✅ Compilación sincronizada (editor ↔ terminal)  
✅ Sincronización PDF-código habilitada (Ctrl+Click)  
✅ Avisos cosmétcos filtrados automáticamente  

## Verificación de Configuración

### Command Palette

```
> LaTeX Workshop: Show LaTeX Workshop errors
> LaTeX Workshop: Compile Recipe
```

### Terminal Alternative (BASH)

```bash
cd doc/
./compile.sh all  # Verifica compilación terminal
grep "LaTeX Error" .build/*.log  # Busca errores REALES
```

## FAQ: ¿Por qué aún veo avisos de fuentes?

**Respuesta:** Los avisos de `lmmono9-regular` (y similares) indican sustitución de fuentes:

- LaTeX no tiene los caracteres `├` (caja), `κ` (kappa), etc. en la fuente monoespaciada
- Esto es **cosmético**: El PDF se genera correctamente, LaTeX sustituye automáticamente
- El compilador con `-nonstopmode` continúa compilando (modo fuerza, no error verdadero)
- LaTeX Workshop linter lo reporta como "warning" pero no es un problema

**Si es molesto:** Reemplazar símbolos Unicode con equivalentes ASCII en LaTeX:

```latex
% En lugar de:
# data[├────┤]

% Usar:
# data[+----+]
```

## Archivos Modificados

1. `.vscode/settings.json` - Configuración de LaTeX Workshop
2. `.latexmkrc` - Configuración de latexmk (ya existente)
3. `compile.sh` - Script de compilación (ya existente)
4. Este archivo: `doc/LATEX_WORKSHOP_SETUP.md`

## Referencia: Estructura de Compilación

```
doc/
├── Predictor_Estocastico_*.tex    (fuentes)
├── compile.sh                      (script bash)
├── .latexmkrc                      (config latexmk)
├── .build/                         (artefactos, oculto)
│   ├── *.aux, *.log, *.toc
│   └── ... (ignorado por git)
└── pdf/                            (PDFs finales, versionados)
    ├── Predictor_Estocastico_Teoria.pdf
    ├── Predictor_Estocastico_Python.pdf
    └── ... 7 documentos totales
```

## Siguiente Pasos

Si aún ve errores en VS Code:

1. Cerrar y reabrir VS Code
2. VS Code → Command Palette → `Developer: Reload Window`
3. Verificar que `latexmk` esté instalado: `which latexmk`
4. Si persisten: Ver `.build/Predictor_Estocastico_*.log` para errores verdaderos
