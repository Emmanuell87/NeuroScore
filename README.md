# NeuroScore: Sistema de Scoring Crediticio mediante Redes Neuronales Artificiales

Este repositorio contiene el desarrollo de un sistema integral de evaluación de riesgo crediticio, utilizando técnicas avanzadas de procesamiento de datos y modelos de aprendizaje profundo (Deep Learning). El proyecto incluye desde el análisis exploratorio y tratamiento de datos hasta el despliegue de una aplicación interactiva para la toma de decisiones.

---

## Información del Proyecto

**Institución:** Universidad Nacional de Colombia  
**Sede:** Medellín.  
**Asignatura:** Redes Neuronales y Algoritmos Bioinspirados  
**Semestre:** 2026-I

### Integrantes

- **Jean Carlos Perilla Garcia** - [jperillag@unal.edu.co](mailto:jperillag@unal.edu.co)
- **Juan Camilo Lopez Morales** - [jlopezmor@unal.edu.co](mailto:jlopezmor@unal.edu.co)
- **Emmanuel Alberto Mejia Arango** - [emmejiaa@unal.edu.co](mailto:emmejiaa@unal.edu.co)

---

## Objetivo del Proyecto

Desarrollar un modelo predictivo capaz de estimar la Probabilidad de Incumplimiento (Probability of Default - PD) de solicitantes de crédito, transformando estas probabilidades en un _Score Crediticio_ interpretable que facilite la clasificación de clientes en bandas de riesgo.

---

## Metodología y Modelado

El flujo de trabajo se dividió en tres fases principales de modelamiento para asegurar robustez y comparabilidad:

1.  **Modelo de Baja Complejidad (Baseline):** Se implementó una **Regresión Logística** con pesos balanceados para establecer una línea base de rendimiento, cumpliendo con los estándares de validación de modelos financieros.
2.  **Modelo Principal (RNA):** Una **Red Neuronal Artificial** optimizada mediante búsqueda de hiperparámetros (`keras_tuner`), evaluando múltiples arquitecturas, tasas de aprendizaje y técnicas de regularización (Dropout, L2).
3.  **Calibración y Scorecard:** Aplicación de **Regresión Isotónica** para calibrar las probabilidades y conversión a puntajes mediante una escala logarítmica (PDO=50, Score 600 a Odds 20:1).

---

## Instalación y Ejecución

### Requisitos Previos

- Python 3.10+
- Entorno virtual configurado (recomendado)
- Dependencias listadas en `requirements.txt`

### Pasos para Ejecutar

1.  **Entrenamiento y Evaluación:**
    Ejecute el script principal para entrenar la RNA y generar los artefactos:
    ```bash
    python codigo.py
    ```
2.  **Evaluación de Baja Complejidad:**
    Para obtener métricas detallas y tablas de confusión del modelo base sin re-entrenar la RNA:
    ```bash
    python evaluar_modelo_baja_complejidad.py
    ```
3.  **Aplicación Web (Streamlit):**
    Visualice el modelo en acción y realice predicciones en tiempo real:
    ```bash
    streamlit run app.py
    ```

---

## Estructura del Repositorio

- `codigo.py`: Script principal de procesamiento, entrenamiento de RNA y visualización de curvas ROC.
- `evaluar_modelo_baja_complejidad.py`: Módulo de evaluación rápida del modelo base (Regresión Logística).
- `app.py`: Aplicación interactiva de Streamlit para la evaluación de perfiles de clientes.
- `artifacts/`: Modelos guardados (`.h5`), escaladores (`.pkl`) y parámetros de calibración.
- `Figure_1.png`: Comparativa visual de las curvas ROC (RNA vs Baseline).

---

## Notas de Revisión

- **Población de Referencia:** Los percentiles y clasificaciones de riesgo mostrados en la aplicación son relativos a la población utilizada para el entrenamiento (Dataset de Créditos).
- **Resultados:** El modelo RNA demostró una mejora significativa en el AUC-ROC frente al modelo de baja complejidad, validando el uso de técnicas de Deep Learning para esta tarea.

---

© 2026 - Universidad Nacional de Colombia
