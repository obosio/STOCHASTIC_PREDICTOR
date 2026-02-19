"""API Layer - Exposure Tier

Facade for external clients and high-level API.

Responsibilities:
  - Public interface definition
  - Configuration loading
  - Load shedding and rate limiting
  - Request/response serialization

See: doc/Predictor_Estocastico_Python.tex §2 - Capa de Exposición

References:
  - API design: doc/Predictor_Estocastico_API_Python.tex
  - Grace period CUSUM: doc/Predictor_Estocastico_API_Python.tex §3.2

Implementation prerequisites:
  - Understand spec for high-level API contract
  - Coordinate with core/ for orchestration
"""
