## Mejora de auditoría de datos sintéticos

El módulo `pfai.privacy.audit` reporta KS y error de correlación para variables numéricas,
distancia de variación total para variables categóricas, utilidad TSTR y duplicados.
La AUC de inferencia de membresía es una prueba empírica heurística, no una garantía de
privacidad: un resultado cercano a 0.5 solo indica que ese ataque concreto no separó bien
los conjuntos. Para una garantía formal, usa entrenamiento con privacidad diferencial y
registra el presupuesto `(ε, δ)` con un contador de privacidad.

Ejecuta las pruebas desde este directorio con `PYTHONPATH=. pytest tests`.
