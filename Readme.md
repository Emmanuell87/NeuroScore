# Credit Risk Scorecard

Este proyecto contiene un modelo de red neuronal para estimar probabilidad de incumplimiento, una scorecard y una aplicación web en Streamlit.

## Ejecución

- La app se ejecuta con Streamlit usando los artefactos ya generados.
- El archivo `loan/loan.csv` se usó para entrenar el modelo y para crear los artefactos precomputados.
- Para que la app inicie más rápido, no vuelve a leer ese CSV pesado al arrancar: usa `reference_defaults.pkl` y `demo_profiles.pkl`.

## Si se reentrena el modelo

Después de reentrenar, vuelve a generar estos archivos para mantener la app alineada con el modelo:

- `modelo_nn_credit_risk.h5`
- `scaler_nn.pkl`
- `label_encoders_nn.pkl`
- `feature_names_nn.pkl`
- `scorecard_params.pkl`
- `pd_calibrator.pkl`
- `scorecard_poblacion.csv`
- `scorecard_resumen_deciles.csv`
- `analisis_variables_riesgo.csv`
- `reference_defaults.pkl`
- `demo_profiles.pkl`

## Nota para revisión

Las clasificaciones y percentiles que muestra la app son relativos a la población del modelo, no reglas fijas de negocio.
