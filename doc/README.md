# Documentación LaTeX - Predictor Estocástico Universal

Este directorio contiene la documentación técnica completa del Predictor Estocástico Universal en formato LaTeX.

## 📁 Estructura de Directorios

```bash
doc/
├── *.tex                           # Archivos fuente LaTeX (7 documentos)
├── pdf/                            # PDFs compilados (versionados en git)
├── .build/                         # Artefactos de compilación (oculto, ignorado por git)
├── .latexmkrc                      # Configuración de compilación (lualatex)
├── compile.sh                      # Script inteligente de compilación bash
└── README.md                       # Este archivo
```

## 📄 Documentos Disponibles

### Documentos Teóricos y Generales

- **Predictor_Estocastico_Teoria.tex** (500+ líneas) - Fundamentos matemáticos, teoremas, **esquemas SDE adaptativos** con transición dinámica Euler/implícito
- **Predictor_Estocastico_Implementacion.tex** (800+ líneas) - Algoritmos, métodos numéricos, **dinámica de Sinkhorn acoplada a volatilidad**
- **Predictor_Estocastico_Pruebas.tex** - Protocolo de validación y pruebas (agnóstico de lenguaje)
- **Predictor_Estocastico_IO.tex** - Especificación de I/O y telemetría

### Documentos Específicos de Python/JAX

- **Predictor_Estocastico_Python.tex** (3000+ líneas) - Guía de implementación en Python con JAX:
  - **Stack tecnológico grabado en piedra**: Justificación rigurosa de Equinox/Diffrax (§1)
  - Optimizaciones de grafo con stop_gradient
  - Implementación completa de 4 ramas (A, B, C, D)
- **Predictor_Estocastico_API_Python.tex** (685+ líneas) - Especificación de API Python, **período de gracia CUSUM** post-cambio de régimen
- **Predictor_Estocastico_Tests_Python.tex** - Suite de pruebas en Python/pytest

## ✨ Mejoras Recientes (Febrero 2026)

| Mejora | Impacto | Documento |
| -------- | --------- | --------- |
| **Stack Equinox/Diffrax grabado en piedra** | Justificación técnica rigurosa (~250 líneas) | Python.tex §1 |
| Transición dinámica SDE (Euler ↔ implícito) | Robustez numérica bajo high stiffness | Teoria.tex §2.3.3 |
| Sinkhorn acoplado a volatilidad | Paisaje suave durante crisis | Implementacion.tex §2.4 |
| Período de gracia CUSUM | Evita cascadas de falsas alarmas | API_Python.tex §3.2 |
| Stop gradient en SIA/CUSUM | Ahorro 30-50% VRAM, 20-40% JIT | Python.tex §3.1 |
| Compilación inteligente | Detecta cambios por timestamps | compile.sh |

## 🚀 Compilación

### Sin argumentos (muestra ayuda por defecto)

```bash
./compile.sh
```

### Compilar solo documentos con cambios

```bash
./compile.sh --all
```

Esto verifica timestamps: solo compila si `.tex` es más nuevo que su `.pdf` correspondiente.

### Forzar recompilación de todos los documentos

```bash
./compile.sh --all --force
# O versión corta:
./compile.sh -a -f
```

Útil cuando necesitas actualizar índices, referencias cruzadas o después de cambios globales.

### Compilar un documento específico

```bash
./compile.sh Predictor_Estocastico_Teoria
# O con extensión:
./compile.sh Predictor_Estocastico_Teoria.tex
```

### Limpiar artefactos de compilación

```bash
./compile.sh clean
```

## 🧠 Cómo Funciona el Script

### Detección Inteligente de Cambios

El script `compile.sh` compara timestamps automáticamente:

```bash
# Estructura interna (simplificada):
if [ "$tex_file" -nt "$pdf_file" ]; then
    compile_doc "$tex_file"  # .tex más nuevo→recompila
else
    echo "⏭️  Sin cambios, omitiendo..."
fi
```

**Beneficios:**

- ⏱️ Compilaciones rápidas cuando nada cambió
- 🎯 Precisión: solo recompila lo necesario
- 📊 Resumen al final: cuántos compilados vs omitidos

### Compilación en Dos Pasadas

Cada documento se compila **dos veces automáticamente** para garantizar convergencia de referencias:

1. **Primera pasada**: Genera archivo `.aux` con etiquetas de referencias
2. **Segunda pasada**: Resuelve referencias cruzadas, actualiza tabla de contenidos, índices

Esto asegura que:

- ✅ Tabla de contenidos sincronizada
- ✅ Referencias cruzadas correctas
- ✅ Números de página actualizados
- ✅ Índices coherentes

### Manejo de Errores

Si hay error de compilación LaTeX:

```bash
🔴 ERRORES ENCONTRADOS EN Predictor_Estocastico_Python.tex:
─────────────────────────────────────────
Predictor_Estocastico_Python.tex:666: error message here
─────────────────────────────────────────
📋 Log completo disponible en:
   doc/.build/Predictor_Estocastico_Python.log
```

El script extrae líneas de error relevantes y proporciona la ruta del log completo para debugging.

## 🎯 Configuración de Compilación

El archivo `.latexmkrc` configura automáticamente:

- **Compilador**: `lualatex` (LuaTeX con soporte Unicode completo)
- **Modo PDF**: `$pdf_mode = 4` (lualatex directo)
- **Directorio de artefactos**: `.build/` (oculto, ignorado por git)
- **Directorio de salida**: `pdf/` (PDFs finales, versionados)
- **Helpers**: `synctex` habilitado para edición inversa

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

- `babel[spanish]` - Soporte para español
- `fontspec` - Gestión de fuentes OpenType
- `amsmath`, `amssymb`, `amsthm` - Matemáticas
- `listings`, `xcolor` - Resaltado de código
- `hyperref` - Enlaces e índices
- `geometry`, `booktabs` - Layout

Instalación automática:

```bash
sudo tlmgr install babel fontspec amsmath amssymb amsthm listings xcolor hyperref geometry booktabs enumitem
```

## 📝 Flujo de Trabajo Dev

### Ciclo Típico

1. **Editar** archivos `.tex` en el editor
2. **Compilar** con `./compile.sh --all` (solo compila cambios)
3. **Revisar** PDFs en `pdf/` (abrir en reader)
4. **Commit** cuando esté listo:

   ```bash
   git add doc/*.tex doc/pdf/*.pdf
   git commit -m "docs: descripción de cambios"
   ```

### Después de Cambios Globales

```bash
# Fuerza recompilación de todo para sincronizar referencias
./compile.sh --all --force
```

### Limpiar y Recompilar (Rebuild Completo)

```bash
./compile.sh clean              # Elimina .build/ y pdf/
./compile.sh --all --force      # Recompila todo desde cero
```

## 📊 Estado Actual (Febrero 2026)

**Últimas mejoras documentadas:**

✅ **Rama C - Esquemas SDE Adaptativos** (Predictor_Estocastico_Teoria.tex)

- Detección automática de rigidez (stiffness) del proceso
- Transición dinámica: Euler explícito → Moulton implícito
- Métrica de rigidez normalizada con umbrales adaptativos
- Esquema híbrido convexo para regímenes intermedios
- Teorema de convergencia fuerte adaptativa

✅ **Transición Dinámica de Sinkhorn** (Predictor_Estocastico_Implementacion.tex)

- Acoplamiento volatilidad-entropía: ε_t = ε₀·(1 + α·σ_t)
- Dinámica suave vs fallback discreto
- Parámetros calibrados para crisis de mercado

✅ **Stack Equinox/Diffrax Grabado en Piedra** (Predictor_Estocastico_Python.tex)

- Justificación técnica rigurosa de cada librería (~250 líneas)
- JAX 0.4.20: Motor XLA con AD y vmap
- Equinox 0.11.3: Framework neuronal para Rama B (DGM) y Rama C (Neural ODEs)
- Diffrax 0.4.1: Solver SDE/ODE diferenciable para Rama C
- Signax 0.1.4: Log-signatures GPU-nativas para Rama D
- OTT-JAX 0.4.5: Transporte óptimo para Orquestador JKO
- Conclusión explícita: "Por lo tanto, este stack está **grabado en piedra** en el diseño del predictor"

✅ **Período de Gracia CUSUM** (Predictor_Estocastico_API_Python.tex)

- Ventana refractoria post-cambio de régimen (10-60 pasos)
- Previene cascadas de falsas alarmas
- Telemetría: monitoreo de G+ durante gracia

✅ **Script de Compilación Mejora** (compile.sh)

- Detección automática de cambios en .tex
- Compilación en dos pasadas (referencias convergentes)
- Forzamiento opcional con --force
- Help por defecto sin argumentos
- Mensajes de error detallados con líneas de problema
- Resumen final: compilados vs omitidos

**Documentos compilados:**

- ✅ Predictor_Estocastico_Teoria.pdf (242 KB, 500+ líneas nuevas)
- ✅ Predictor_Estocastico_Implementacion.pdf (233 KB)
- ✅ Predictor_Estocastico_API_Python.pdf (215 KB)
- ✅ Predictor_Estocastico_IO.pdf (169 KB)
- ✅ Predictor_Estocastico_Python.pdf (470 KB, **stack grabado en piedra** ~250 líneas)
- ✅ Predictor_Estocastico_Tests_Python.pdf (295 KB)
- ✅ Predictor_Estocastico_Pruebas.pdf (267 KB)

**Total:** 1.73 MB documentación sincronizada

### Tabla de Características Documentadas

| Feature | Status | Documento | Beneficio |
| --------- | -------- | --------- | ---------- |
| **Stack Grabado en Piedra** | ✅ | Python.tex §1 | Rigor arquitectónico |
| Esquemas SDE Dinámicos | ✅ | Teoria.tex | Robustez numérica |
| Sinkhorn Acoplado Volatilidad | ✅ | Implementacion.tex | Crisis-proof |
| Período Gracia CUSUM | ✅ | API_Python.tex | Anti-cascadas |
| Stop Gradient JAX | ✅ | Python.tex | Eficiencia VRAM/JIT |
| Compilación Inteligente | ✅ | compile.sh | Dev speed |

## ✨ Ventajas de Esta Configuración

- ✅ **Workspace limpio**: Solo archivos fuente visibles (artefactos en `.build/` oculto)
- ✅ **Compilación inteligente**: Detecta cambios automáticamente
- ✅ **Índices actualizados**: Dos pasadas garantizan convergencia
- ✅ **Errores claros**: Script muestra líneas problemáticas
- ✅ **Git amigable**: Artefactos no contaminan historial; solo PDFs versionados
- ✅ **Reproducible**: Configuración versionada en `.latexmkrc` y `compile.sh`
- ✅ **LuaTeX moderno**: Soporte Unicode, fuentes OpenType, características avanzadas

## 🔧 Configuración del Editor

### VS Code (sin extensiones necesarias)

Configurar `.vscode/settings.json`:

```json
{
  "files.exclude": {
    "**/.*": true,
    "**/__pycache__": true
  },
  "[latex]": {
    "editor.formatOnSave": false
  }
}
```

### Editor Local + Terminal

Usar `./compile.sh` directamente desde terminal:

```bash
cd doc
./compile.sh --all            # Compila solo cambios
# Luego abrir PDFs en pdf/ con tu reader favorito
```

## ⚠️ Avisos de Compilación Conocidos

Se reportan advertencias menores sobre caracteres faltantes en fuentes monoespaciadas:

- Símbolos griegos (κ, γ, ρ) en bloques de código
- Caracteres especiales de caja de dibujo (├, ─, etc.)

**Impacto**: Cosmético. Los PDFs se generan completamente sin errores; las advertencias solo indican sustituciones de fuentes en entornos monoespaciados.

**Solución** (si es necesario): Usar fuentes Unicode o replacer símbolos con comandos LaTeX equivalentes.

## 📚 Referencias

- [latexmk documentation](https://mg.readthedocs.io/latexmk.html)
- [LaTeX project](https://www.latex-project.org/)
- [LuaTeX documentation](http://www.luatex.org/)
- [fontspec package](https://ctan.org/pkg/fontspec)
