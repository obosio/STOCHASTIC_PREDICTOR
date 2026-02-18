#!/usr/bin/env python
"""
Script de verificación del entorno de desarrollo.
Verifica que todas las dependencias críticas estén instaladas correctamente.
"""

import sys

def check_module(name, min_version=None):
    """Verifica que un módulo esté instalado y opcionalmente su versión."""
    try:
        module = __import__(name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {name:20s} {version}")
        return True
    except ImportError:
        print(f"❌ {name:20s} NOT INSTALLED")
        return False

def main():
    print("=" * 60)
    print("STOCHASTIC PREDICTOR - Verificación de Entorno")
    print("=" * 60)
    print()
    
    # Verificar Python
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print()
    
    # Paquetes críticos
    print("Dependencias Principales:")
    print("-" * 60)
    critical_packages = [
        'jax',
        'jaxlib',
        'equinox',
        'diffrax',
        'jaxtyping',
        'ott',
        'pywt',  # PyWavelets
        'scipy',
        'signax',
        'numpy',
        'pandas',
    ]
    
    all_ok = True
    for pkg in critical_packages:
        if not check_module(pkg):
            all_ok = False
    
    print()
    print("Herramientas de Desarrollo:")
    print("-" * 60)
    dev_packages = [
        'pytest',
        'black',
        'flake8',
        'mypy',
        'isort',
    ]
    
    for pkg in dev_packages:
        check_module(pkg)
    
    print()
    print("Visualización:")
    print("-" * 60)
    viz_packages = [
        'matplotlib',
        'seaborn',
    ]
    
    for pkg in viz_packages:
        check_module(pkg)
    
    print()
    print("=" * 60)
    
    # Test de JAX
    print("\nTest de Funcionalidad de JAX:")
    print("-" * 60)
    try:
        import jax
        import jax.numpy as jnp
        
        # Test básico
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sin(x)
        print(f"✅ JAX computation test: sin([1, 2, 3]) = {y}")
        
        # Verificar dispositivos
        devices = jax.devices()
        print(f"✅ JAX devices available: {devices}")
        
        # Test de JIT
        @jax.jit
        def f(x):
            return x ** 2 + 2 * x + 1
        
        result = f(jnp.array([1.0, 2.0, 3.0]))
        print(f"✅ JAX JIT compilation test: (x² + 2x + 1) for x=[1,2,3] = {result}")
        
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✅ ¡Entorno configurado correctamente!")
        print("Listo para comenzar Fase 0 del desarrollo")
        return 0
    else:
        print("⚠️  Hay paquetes faltantes. Por favor instala las dependencias.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
