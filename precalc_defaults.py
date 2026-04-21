"""
Ejecutar este script UNA VEZ después de entrenar (codigo.py) para generar:
  - defaults_nn.pkl     → valores de referencia (medianas/modas) para la app
  - demo_profiles_nn.pkl → 3 casos de ejemplo reales para la app

Estos archivos reemplazan la necesidad de loan/loan.csv en producción.
La lógica aquí es idéntica a la de build_reference_values y
build_full_demo_profiles en app.py.
"""

import numpy as np
import pandas as pd
import joblib

print("[INFO] Cargando artefactos del modelo...")
label_encoders = joblib.load("label_encoders_nn.pkl")
feature_names = joblib.load("feature_names_nn.pkl")
cat_features = list(label_encoders.keys())

print("[INFO] Cargando loan/loan.csv...")
loan = pd.read_csv("loan/loan.csv", low_memory=False)

# --- 1. defaults_nn.pkl ---
# Misma lógica que build_reference_values en app.py
print("[INFO] Calculando valores de referencia...")
defaults = {}
for col in feature_names:
    if col in cat_features:
        if col in loan.columns:
            mode_series = loan[col].dropna().astype(str)
            defaults[col] = mode_series.mode().iloc[0] if not mode_series.empty else "Missing"
        else:
            defaults[col] = "Missing"
    else:
        if col in loan.columns:
            num = pd.to_numeric(loan[col], errors="coerce")
            med = num.median()
            defaults[col] = float(med) if pd.notna(med) else 0.0
        else:
            defaults[col] = 0.0

joblib.dump(defaults, "defaults_nn.pkl")
print("✓ Guardado: defaults_nn.pkl")

# --- 2. demo_profiles_nn.pkl ---
# Misma lógica que build_full_demo_profiles en app.py
print("[INFO] Calculando perfiles demo...")

def row_to_profile(row):
    profile = {}
    for col in feature_names:
        if col in cat_features:
            profile[col] = str(row[col]) if pd.notna(row[col]) else "Missing"
        else:
            value = pd.to_numeric(row[col], errors="coerce") if col in row.index else np.nan
            profile[col] = float(value) if pd.notna(value) else 0.0
    return profile

scored = pd.read_csv("scorecard_poblacion.csv")

low_idx = scored[scored["target"] == 0]["pd_bad"].idxmin()
med_idx = (scored["pd_bad"] - scored["pd_bad"].median()).abs().idxmin()
high_idx = scored[scored["target"] == 1]["pd_bad"].idxmax()

demo_profiles = {
    "Caso real - Bajo riesgo": row_to_profile(loan.loc[low_idx]),
    "Caso real - Riesgo medio": row_to_profile(loan.loc[med_idx]),
    "Caso real - Alto riesgo": row_to_profile(loan.loc[high_idx]),
}

joblib.dump(demo_profiles, "demo_profiles_nn.pkl")
print("✓ Guardado: demo_profiles_nn.pkl")
print("\n[LISTO] Ahora puedes subir a GitHub sin loan/loan.csv")
