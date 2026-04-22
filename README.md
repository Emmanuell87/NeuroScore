# Modelo de Riesgo de Crédito con Red Neuronal

## Descripción General

Este proyecto implementa un sistema completo de evaluación de riesgo de crédito basado en redes neuronales artificiales. El sistema incluye:

- **Modelo predictivo**: Red neuronal que estima probabilidad de incumplimiento (PD)
- **Scorecard**: Conversión de PD a score interpretable (0-900)
- **Análisis de variables**: Identificación de factores de riesgo
- **Aplicación web**: Interfaz Streamlit para consultar riesgo por perfil

---

## Estructura del Proyecto

```
Redes_neuronales/
├── codigo.py                        # Script de entrenamiento del modelo
├── app.py                           # Aplicación web Streamlit
├── Perfiles_prueba.py              # Generador de perfiles demo
├── loan/
│   └── loan.csv                    # Dataset de entrenamiento (historco)
├── modelo_nn_credit_risk.h5        # Modelo entrenado (Keras/TensorFlow)
├── scaler_nn.pkl                   # Escalador Standard para variables numéricas
├── label_encoders_nn.pkl           # Codificadores de variables categóricas
├── feature_names_nn.pkl            # Lista de 48 features del modelo
├── reference_defaults.pkl          # Valores de referencia para UI
├── demo_profiles.pkl               # Perfiles de ejemplo (bajo/medio/alto riesgo)
├── pd_calibrator.pkl               # Calibrador isotónico de probabilidades
├── scorecard_params.pkl            # Parámetros A, B de scorecard
├── scorecard_poblacion.csv         # PD y score de todo el dataset
├── scorecard_resumen_deciles.csv   # Resumen por deciles de riesgo
├── analisis_variables_riesgo.csv   # Variables ordenadas por importancia
└── requirements.txt                # Dependencias Python
```

---

## 🔄 Flujo de Funcionamiento

### **Fase 1: Entrenamiento (codigo.py)**

1. **Carga de datos**: Lee `loan/loan.csv`
2. **Preprocesamiento**:
   - Limpia valores faltantes
   - Separa features numéricas (18) y categóricas (30)
   - Codifica categóricas con `LabelEncoder`
   - Escala numéricas con `StandardScaler`
3. **División**: Train/Validation/Test
4. **Entrenamiento de Red Neuronal**:
   - Arquitectura: 2-3 capas densas con dropout
   - Ajuste de hiperparámetros con `Keras Tuner`
   - Validación con métricas: AUC, F1, Precisión, Recall
5. **Calibración**: Aplica calibración isotónica para que la salida sea PD confiable
6. **Generación de Scorecard**:
   - Calcula parámetros A y B para transformar PD → Score
   - Crea tabla poblacional por deciles
7. **Análisis de Variables**: Importancia y dirección de riesgo
8. **Guardado de Artefactos**: Todos los `.pkl` y `.csv` necesarios

**Salida**: Modelo listo para inferencia + archivos de consulta

---

### **Fase 2: Inferencia (app.py)**

1. **Carga ligera**: Importa artefactos preentrenados (sin leer CSV)
2. **Entrada de usuario**: Formulario Streamlit en modo básico o avanzado
3. **Preprocesamiento**:
   - Aplica mismo escalado y codificación que en entrenamiento
4. **Predicción**:
   - Red neuronal → PD bruto
   - Calibrador → PD ajustado
   - Scorecard → Score crediticio
5. **Comparación poblacional**:
   - Computa percentil: ¿qué % tiene PD ≤ que el cliente?
6. **Visualización**:
   - Banda de riesgo (Muy bajo, Bajo, Medio, Alto, Muy alto)
   - Comparativa con deciles
   - Explicación de variables influyentes

**Entrada**: 48 features del cliente  
**Salida**: PD, Score, Banda de riesgo, Percentil

---

## Componentes Principales

### **1. Modelo Neural (`modelo_nn_credit_risk.h5`)**

- Entrada: 48 features preprocesadas
- Capas ocultas: ~100-200 neuronas con dropout
- Salida: Probabilidad bruta de incumplimiento (0-1)
- Función de pérdida: Binary Crossentropy
- Optimizador: Adam

### **2. Escalador (`scaler_nn.pkl`)**

- Transforma variables numéricas a media=0, desv=1
- Aplicado en entrenamiento Y en cada predicción
- Mantiene la misma escala entre train e inferencia

### **3. Encoders (`label_encoders_nn.pkl`)**

- Un encoder por variable categórica
- Transforma strings (ej. "A", "RENT") a índices numéricos
- Consistencia: si aparece valor no visto, mapea a "Missing"

### **4. Calibrador (`pd_calibrator.pkl`)**

- Calibración isotónica entrenada en validación
- Transforma PD bruta → PD calibrada más confiable
- Asegura que predicciones sean probabilities reales

### **5. Parámetros Scorecard (`scorecard_params.pkl`)**

Contiene:

```python
{
  'A': valor_base,           # Intercept del score
  'B': pendiente,            # Pendiente del log-odds
  'PDO': 50,                 # Puntos en doblar odds (estándar 50)
  'score_at_odds': 600,      # Score en odds referencia
  'odds_ref': valor          # Odds de referencia
}
```

Fórmula: `Score = A + B × ln(odds)` donde `odds = (1-PD)/PD`

### **6. Población (`scorecard_poblacion.csv`)**

Tabla con PD y Score de cada registro de entrenamiento. Usada para:

- Comparar percentil del usuario
- Mostrar distribución
- Crear deciles

### **7. Perfiles Demo (`demo_profiles.pkl`)**

Diccionario con 4 casos de ejemplo:

- Bajo riesgo: ingreso alto, DTI bajo, delinquencias=0
- Riesgo medio: ingreso medio, DTI medio, algún delinquencia
- Alto riesgo: ingreso bajo, DTI alto, múltiples delinquencias
- Típico: todos los valores en mediana

Usados para demostración en la app sin necesidad de Dataset

---

## Cómo Usar

### **Ejecutar la app**

```bash
streamlit run app.py
```

La app abrirá en navegador (http://localhost:8501) y permitirá:

1. Seleccionar modo: Básico (recomendado) o Avanzado
2. Elegir perfil demo o ingresar valores manuales
3. Ver predicción: PD, Score, Banda, Percentil
4. Explorar análisis de variables

### **Reentrenar el modelo**

```bash
python codigo.py
```

Esto regenera todos los artefactos. Si cambias hiperparámetros, el modelo new se guardará automáticamente.

### **Regenerar perfiles demo** (opcional, después de reentrenar)

```bash
python Perfiles_prueba.py
```

Crea 4 nuevos perfiles de prueba alineados con el nuevo modelo.

---

## Archivos de Salida y Su Rol

| Archivo                         | Propósito                 | Actualización               |
| ------------------------------- | ------------------------- | --------------------------- |
| `modelo_nn_credit_risk.h5`      | Pesos de la red           | Al entrenar                 |
| `scaler_nn.pkl`                 | Escalador numéricas       | Al entrenar                 |
| `label_encoders_nn.pkl`         | Codificadores categóricas | Al entrenar                 |
| `feature_names_nn.pkl`          | Lista de 48 features      | Raro cambio                 |
| `pd_calibrator.pkl`             | Calibrador isotónico      | Al entrenar                 |
| `scorecard_params.pkl`          | Parámetros A, B, etc.     | Al entrenar                 |
| `reference_defaults.pkl`        | Valores por defecto UI    | Al entrenar                 |
| `demo_profiles.pkl`             | Perfiles de ejemplo       | Ejecutar Perfiles_prueba.py |
| `scorecard_poblacion.csv`       | PD/Score de población     | Al entrenar                 |
| `scorecard_resumen_deciles.csv` | Resumen por deciles       | Al entrenar                 |
| `analisis_variables_riesgo.csv` | Variables ordenadas       | Al entrenar                 |

---

## Dependencias

```
tensorflow>=2.10
keras>=2.10
scikit-learn>=1.0
pandas>=1.3
numpy>=1.20
streamlit>=1.20
joblib>=1.2
matplotlib>=3.5
seaborn>=0.12
```

Instalar:

```bash
pip install -r requirements.txt
```

---

## Flujo Técnico: Un Ejemplo

### Cliente entra a la app e ingresa:

- `annual_inc`: 50000
- `dti`: 22
- `grade`: 'C'
- (resto en valores default)

### Proceso interno:

1. **Preprocesamiento**:
   - `annual_inc` y `dti` → escalar con media/std del training
   - `grade='C'` → buscar en encoder, convertir a índice (ej. 5)

2. **Predicción**:
   - Pasar array preprocesado a la red neuronal
   - Obtiene PD bruto: 0.25 (ejemplo)

3. **Calibración**:
   - Pasar 0.25 al calibrador isotónico
   - Obtiene PD calibrada: 0.23

4. **Scorecard**:
   - Calcular `odds = (1 - 0.23) / 0.23 = 3.35`
   - `Score = A + B × ln(3.35)`
   - Resultado: 580 (ejemplo)

5. **Comparativa**:
   - Contar cuántos en población tienen PD ≤ 0.23
   - Resultado: 35% → percentil 35
   - Clasificación: "Riesgo Medio"

6. **Visualización**:
   - Mostrar Score 580, PD 23%, Percentil 35, Banda Riesgo Medio
   - Listar variables que más impactaron

---

## Consideraciones Importantes

- **No usa loan.csv en runtime**: La app carga solo artefactos pkl/csv, por eso es ligera
- **Valores default**: Si el usuario no especifica una variable, se usa mediana de training
- **Percentiles relativos**: No son umbrales de negocio, sino posición en población
- **Reentrenamiento**: Cambia todos los artefactos, la app se sincroniza automáticamente
- **Perfiles demo**: Se actualizan manualmente con `Perfiles_prueba.py` si es necesario

---

## Uso Práctico

**Caso de uso**: Evaluador de riesgos de un banco necesita estimar PD de un solicitante

1. Abre la app con `streamlit run app.py`
2. Ingresa datos del cliente en modo básico
3. Obtiene automáticamente: probabilidad de incumplimiento, score, banda de riesgo
4. Compara contra población usando percentil
5. Revisa qué variables más impactan su riesgo

---

**Versión**: 1.0 | **Última actualización**: 2026-04-21
