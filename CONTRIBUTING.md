# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto Universal Stochastic Predictor (USP)!

## 🚀 Cómo Contribuir

### Reportar Issues

- Usa el sistema de issues de GitHub para reportar bugs o sugerir features
- Describe claramente el problema o la sugerencia
- Incluye pasos para reproducir el bug si es aplicable
- Menciona tu entorno (versión de Python, JAX, sistema operativo)

### Proceso de Pull Request

1. **Fork** el repositorio
2. **Crea una rama** para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre un Pull Request**

### Estándares de Código

#### Python

- Sigue [PEP 8](https://pep8.org/)
- Usa type hints (tipado estático con `jaxtyping`)
- Documenta funciones con docstrings estilo Google
- Mantén las funciones puras cuando sea posible (JAX requirement)

#### Documentación

- La documentación técnica se escribe en LaTeX
- Los comentarios de código deben ser claros y en español o inglés
- Actualiza el README si añades nuevas funcionalidades

### Testing


- Todos los PRs deben incluir tests unitarios
- Usa `pytest` para los tests
- Asegúrate de que todos los tests pasen antes de hacer el PR

### Estructura de Commits

Usa mensajes de commit descriptivos:

```text
feat: Implementa núcleo de predicción tipo A (RKHS)
fix: Corrige bug en estimación WTMM
docs: Actualiza documentación de API
test: Añade tests para orquestador JKO
```

## 📋 Áreas de Contribución

### Prioridad Alta

- [ ] Implementación del motor SIA/WTMM
- [ ] Desarrollo de núcleos de predicción
- [ ] Sistema de tests y benchmarks
- [ ] Ejemplos de uso

### Prioridad Media

- [ ] Optimizaciones de rendimiento
- [ ] Documentación adicional
- [ ] Visualizaciones y dashboards

### Prioridad Baja

- [ ] Integraciones con otras librerías
- [ ] Soporte para nuevos backends

## 🤝 Código de Conducta

### Nuestro Compromiso

- Mantener un ambiente acogedor e inclusivo
- Respetar diferentes puntos de vista y experiencias
- Aceptar críticas constructivas con gracia
- Enfocarse en lo mejor para la comunidad

### Comportamiento Esperado

- Usar lenguaje acogedor e inclusivo
- Respetar diferentes puntos de vista
- Aceptar críticas constructivas
- Mostrar empatía hacia otros miembros

### Comportamiento Inaceptable

- Lenguaje o imágenes sexualizadas
- Trolling, insultos o ataques personales
- Acoso público o privado
- Publicar información privada de otros sin permiso

## 📞 Contacto

Si tienes preguntas sobre cómo contribuir, abre un issue con la etiqueta `question`.

## 🙏 Reconocimientos

Todos los contribuidores serán reconocidos en el proyecto. ¡Gracias por ayudar a mejorar USP!
