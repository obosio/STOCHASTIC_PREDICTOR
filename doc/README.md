# Especificación Técnica Completa - Predictor Estocástico Universal

Este directorio contiene la **especificación matemática y algorítmica completa** del Predictor Estocástico Universal en formato LaTeX (3000+ líneas, 7 documentos, 1.73 MB PDFs).

> ⚠️ **Este es un proyecto de especificación pura**: No incluye código de implementación, solo documentación técnica rigurosa lista para traducción directa a código.

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

## 📄 Especificaciones Disponibles

### Especificaciones Teóricas y Arquitectónicas

- **Predictor_Estocastico_Teoria.tex** (500+ líneas) - Fundamentación matemática rigurosa:
  - Teoría de procesos estocásticos, transporte óptimo, rough paths
  - **Esquemas SDE adaptativos** con transición dinámica Euler/implícito
  - Teoremas de convergencia y análisis de complejidad
- **Predictor_Estocastico_Implementacion.tex** (800+ líneas) - Algoritmos y métodos numéricos:
  - Pseudocódigo detallado independiente de lenguaje
  - **Dinámica de Sinkhorn acoplada a volatilidad**
  - Estrategias de optimización y paralelización
- **Predictor_Estocastico_Pruebas.tex** - Protocolo de validación y casos de prueba
- **Predictor_Estocastico_IO.tex** - Especificación de interfaces de entrada/salida

### Especificaciones Python/JAX (Listas para Implementación)

- **Predictor_Estocastico_Python.tex** (3000+ líneas) - **Especificación completa en Python/JAX**:
  - **§1: Stack grabado en piedra** (~250 líneas): Justificación rigurosa de JAX/Equinox/Diffrax/Signax/OTT-JAX
  - **§2-6: Implementación de 4 ramas**: Pseudocódigo Python completo con tipos JAX
  - **Optimizaciones XLA**: stop_gradient, JIT, vmap, estrategias de compilación
  - **Nivel de detalle**: Traducible directamente a código funcional
- **Predictor_Estocastico_API_Python.tex** (685+ líneas) - Especificación de API:
  - Interfaces públicas, contratos de función, tipos
  - **Período de gracia CUSUM** post-cambio de régimen
  - Telemetría y logging
- **Predictor_Estocastico_Tests_Python.tex** - Suite de tests especificada:
  - Casos de prueba con datos sintéticos
  - Métricas de validación y criterios de aceptación
  - Estrategia pytest + fixtures JAX

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

## 📊 Estado de Especificación (Febrero 2026)

**Especificación completa al 100%** - Lista para implementación futura.

### Componentes Especificados

✅ **Stack Tecnológico (Grabado en Piedra)** - Predictor_Estocastico_Python.tex §1

- Justificación técnica rigurosa de cada librería (~250 líneas)
- **JAX 0.4.20**: Motor XLA con AD, vmap, jit (capa fundamental obligatoria)
- **Equinox 0.11.3**: Framework neuronal pythonico para Rama B (DGM) y Rama C (Neural ODEs)
- **Diffrax 0.4.1**: Solver diferenciable de SDEs/ODEs para Rama C
- **Signax 0.1.4**: Log-signatures GPU-nativas para Rama D
- **OTT-JAX 0.4.5**: Transporte óptimo (Sinkhorn) para Orquestador JKO
- **PyWavelets 1.4.1**: Transformada wavelet para SIA (WTMM)
- Análisis de alternativas descartadas (Flax, Haiku, PyTorch, torchdiffeq)
- Conclusión formal: _"Por lo tanto, este stack está **grabado en piedra** en el diseño del predictor"_

✅ **Esquemas SDE Adaptativos** - Predictor_Estocastico_Teoria.tex §2.3.3

- Detección automática de rigidez (stiffness) del proceso
- Transición dinámica: Euler explícito ↔ Moulton implícito
- Métrica de rigidez normalizada con umbrales adaptativos
- Esquema híbrido convexo para regímenes intermedios
- Teorema de convergencia fuerte adaptativa

✅ **Dinámica de Sinkhorn Acoplada a Volatilidad** - Predictor_Estocastico_Implementacion.tex §2.4

- Acoplamiento volatilidad-entropía: ε_t = ε₀·(1 + α·σ_t)
- Dinámica suave vs fallback discreto
- Parámetros calibrados para crisis de mercado

✅ **Período de Gracia CUSUM** - Predictor_Estocastico_API_Python.tex §3.2

- Ventana refractoria post-cambio de régimen (10-60 pasos)
- Previene cascadas de falsas alarmas
- Telemetría: monitoreo de G+ durante gracia

✅ **Optimización del Grafo XLA** - Predictor_Estocastico_Python.tex §3.1

- Stop gradient en módulos diagnósticos (SIA, CUSUM)
- Ahorro esperado: 30-50% VRAM, 20-40% tiempo JIT
- Backpropagation solo en módulos predictivos

✅ **Build System Inteligente** - compile.sh

- Detección automática de cambios por timestamps
- Compilación en dos pasadas (referencias convergentes)
- Modo --force para recompilación total
- Mensajes de error detallados

### PDFs Compilados (Listos para Implementación)

- ✅ Predictor_Estocastico_Teoria.pdf (242 KB) - Fundamentación matemática
- ✅ Predictor_Estocastico_Implementacion.pdf (233 KB) - Algoritmos detallados
- ✅ Predictor_Estocastico_API_Python.pdf (215 KB) - Especificación de interfaces
- ✅ Predictor_Estocastico_IO.pdf (169 KB) - Entrada/salida del sistema
- ✅ **Predictor_Estocastico_Python.pdf (470 KB)** - Especificación Python/JAX completa
- ✅ Predictor_Estocastico_Tests_Python.pdf (295 KB) - Suite de tests
- ✅ Predictor_Estocastico_Pruebas.pdf (267 KB) - Casos de validación

**Total:** 1.73 MB de especificación técnica completa

### Tabla de Características Documentadas

| Feature | Status | Documento | Beneficio |
| --------- | -------- | --------- | ---------- |
| **Stack Grabado en Piedra** | ✅ | Python.tex §1 | Rigor arquitectónico |
| Esquemas SDE Dinámicos | ✅ | Teoria.tex | Robustez numérica |
| Sinkhorn Acoplado Volatilidad | ✅ | Implementacion.tex | Crisis-proof |
| Período Gracia CUSUM | ✅ | API_Python.tex | Anti-cascadas |
| Stop Gradient JAX | ✅ | Python.tex | Eficiencia VRAM/JIT |
| Compilación Inteligente | ✅ | compile.sh | Dev speed |

## ✨ Ventajas de Esta Especificación

- ✅ **Especificación completa y autocontenida**: 3000+ líneas, 7 documentos, todos los algoritmos detallados
- ✅ **Lista para implementación**: Pseudocódigo Python traducible directamente a código funcional
- ✅ **Stack justificado rigurosamente**: Análisis técnico de JAX/Equinox/Diffrax con alternativas descartadas
- ✅ **Nivel Diamante**: Decisiones arquitectónicas documentadas ANTES de código
- ✅ **Workspace limpio**: Solo especificaciones, sin código de implementación
- ✅ **Compilación inteligente**: Script detecta cambios automáticamente
- ✅ **PDFs versionados**: 1.73 MB de especificaciones compiladas en git
- ✅ **Reproducible**: Configuración LaTeX versionada, builds deterministas
- ✅ **LuaTeX moderno**: Soporte Unicode, ecuaciones complejas, referencias cruzadas

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
