# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto Universal Stochastic Predictor (USP)!

## ⚠️ Alcance de Contribuciones

Este repositorio contiene **únicamente la especificación técnica** (documentos LaTeX). Las contribuciones deben enfocarse en **mejorar, clarificar y extender la especificación**, no en implementar código.

## 🚀 Cómo Contribuir

### Reportar Problemas en la Especificación

- **Issues**: Usa el sistema de issues de GitHub para:
  - ❌ Errores matemáticos
  - ❌ Inconsistencias entre secciones (ej: referencia a variables no definidas)
  - ❌ Ambigüedades o claridades faltantes
  - ❌ Algoritmos que requieren aclaración
  
- **Formato**: Incluye siempre el archivo y sección específica (ej: `Python.tex §3.2`)

### Sugerir Mejoras a la Especificación

- Extensiones algorítmicas justificadas matemáticamente
- Alternativas descartadas con análisis comparativo
- Casos de uso adicionales
- Análisis de complejidad computacional mejorado

### Proceso de Pull Request

1. **Fork** el repositorio
2. **Crea una rama** con nombre descriptivo (`fix/typo-sde` o `enhance/sinkhorn-analysis`)
3. **Edita archivos `.tex`** en el directorio `doc/`
4. **Compila locally** con `./doc/compile.sh` para verificar LaTeX válido
5. **Commit** con mensaje descriptivo:

   ```
   docs: Corrige notación de matriz en Python.tex §2.1
   docs: Amplía análisis WTMM en Teoria.tex §3.3
   docs: Aclara período de gracia CUSUM en API_Python.tex
   ```

6. **Push** y abre un Pull Request con descripción clara de cambios

### Estándares de Especificación

#### LaTeX/Documentación

- ✅ Usar comandos LaTeX consistentes con documentos existentes
- ✅ Mantener estructura de secciones coherente
- ✅ Incluir referencias cruzadas (`\ref{}`, `\cite{}`)
- ✅ Definir notación matemática antes de usarla
- ✅ Incluir ejemplos o pseudocódigo cuando sea posible
- ✅ Traducir a español si estás en doc español; a inglés si en doc inglés
- ✅ Line length ≤ 100 caracteres para mantener legibilidad en git diffs

#### Notación Matemática

- ✅ Use \textbf{} para énfasis
- ✅ Definir espacios ($\mathbb{R}$, $L^2(\Omega)$, $\mathcal{H}$) al introducirlos
- ✅ Usar subíndices consistentes (ej: siempre $X_t$, nunca $X(t)$)
- ✅ Incluir dimensiones cuando sea crítico

## 📋 Áreas de Contribución

### Especificación Base (Prioridad Alta)

- Errores en derivaciones matemáticas
- Inconsistencias de notación
- Referencias cruzadas rotas
- Pseudocódigo que necesita aclaración

### Extensiones Propuestas (Prioridad Media)

- Nuevos kernels de predicción (justificación matemática)
- Alternativas de orquestación adaptativa
- Análisis comparativo con métodos existentes
- Casos de uso especializados

### Mejoras Documentales (Prioridad Baja)

- Diagramas o visualizaciones conceptuales
- Índice mejorado
- Ejemplo adicional de pseudocódigo
- Apéndices con derivaciones detalladas

## 🤝 Código de Conducta

### Nuestro Compromiso

- Ambiente acogedor e inclusivo basado en rigor intelectual
- Respetar diferentes perspectivas matemáticas y de ingeniería
- Aceptar críticas técnicas constructivas
- Enfocarse en calidad e integridad de la especificación

### Comportamiento Esperado

- Usar lenguaje técnico preciso
- Respetar puntos de vista alternativos con justificación
- Aceptar críticas de especificación sin ego
- Mostrar empatía hacia otros revisores

### Comportamiento Inaceptable

- Ataques ad hominem a autores o contribuidores
- Rechazo de cambios válidos sin justificación técnica
- Lenguaje discriminatorio o acoso
- Publicar información privada sin permiso

## 📝 Proceso de Revisión

1. **Sintaxis LaTeX**: El CI automáticamente verifica que la especificación compile
2. **Revisión técnica**: Mantenedores verifican consistencia matemática
3. **Completitud**: ¿Están claros los cambios? ¿Se actualizan referencias cruzadas?
4. **Merge**: Una vez aprobado, se fusiona a `main`

## 📱 Contacto

- **Issues**: Para reportes de especificación específicos
- **Discussions**: Para debates generales sobre arquitectura o algoritmos
- **Email**: Contacta a mantainers si tienes preguntas previas

## 🙏 Reconocimientos

Todos los contribuidores a la especificación serán reconocidos en el archivo [CHANGELOG.md](CHANGELOG.md) y en los commits relevantes.

---

Gracias por ayudar a refinar y mejorar la especificación del Predictor Estocástico Universal. 🚀
