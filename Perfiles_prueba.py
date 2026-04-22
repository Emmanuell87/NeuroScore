# crear_demo_profiles.py
# Ejecutar DESPUÉS de codigo.py para generar los perfiles de prueba

import pandas as pd
import numpy as np
import joblib
import os

print("="*60)
print("CREANDO demo_profiles.pkl")
print("="*60)

# Cargar archivos necesarios
try:
    label_encoders = joblib.load('label_encoders_nn.pkl')
    feature_names = joblib.load('feature_names_nn.pkl')
    reference_defaults = joblib.load('reference_defaults.pkl')
    print(" Archivos base cargados")
except Exception as e:
    print(f" Error: {e}")
    print("Ejecuta primero 'codigo.py' para generar los archivos base")
    exit()

cat_features = list(label_encoders.keys())
num_features = [c for c in feature_names if c not in cat_features]

demo_profiles = {}

# ---------------------------
# PERFIL 1: BAJO RIESGO 
# ---------------------------
print("\n Perfil BAJO RIESGO...")
low_risk = {}
for col in num_features:
    if col == 'annual_inc': low_risk[col] = 120000.0
    elif col == 'dti': low_risk[col] = 5.0
    elif col == 'delinq_2yrs': low_risk[col] = 0
    elif col == 'revol_util': low_risk[col] = 20.0
    elif col == 'loan_amnt': low_risk[col] = 10000.0
    elif col == 'int_rate': low_risk[col] = 6.5
    elif col == 'fico_range_low': low_risk[col] = 750.0
    else: low_risk[col] = float(reference_defaults.get(col, 0))

for col in cat_features:
    if col == 'grade': low_risk[col] = 'A'
    elif col == 'sub_grade': low_risk[col] = 'A1'
    elif col == 'term': low_risk[col] = '36 months'
    elif col == 'home_ownership': low_risk[col] = 'OWN'
    else: low_risk[col] = reference_defaults.get(col, 'Missing')

demo_profiles[" Perfil Bajo Riesgo (Buen Pagador)"] = low_risk

# ---------------------------
# PERFIL 2: RIESGO MEDIO 
# ---------------------------
print(" Perfil RIESGO MEDIO ...")
medium_risk = {}
for col in num_features:
    if col == 'annual_inc': medium_risk[col] = 50000.0
    elif col == 'dti': medium_risk[col] = 22.0
    elif col == 'delinq_2yrs': medium_risk[col] = 1
    elif col == 'revol_util': medium_risk[col] = 70.0
    elif col == 'loan_amnt': medium_risk[col] = 18000.0
    elif col == 'int_rate': medium_risk[col] = 14.0
    elif col == 'fico_range_low': medium_risk[col] = 650.0
    else: medium_risk[col] = float(reference_defaults.get(col, 0))

for col in cat_features:
    if col == 'grade': medium_risk[col] = 'C'
    elif col == 'sub_grade': medium_risk[col] = 'C2'
    elif col == 'term': medium_risk[col] = '36 months'
    elif col == 'home_ownership': medium_risk[col] = 'RENT'
    else: medium_risk[col] = reference_defaults.get(col, 'Missing')

demo_profiles[" Perfil Riesgo Medio "] = medium_risk

# ---------------------------
# PERFIL 3: ALTO RIESGO 
# ---------------------------
print(" Perfil ALTO RIESGO...")
high_risk = {}
for col in num_features:
    if col == 'annual_inc': high_risk[col] = 28000.0
    elif col == 'dti': high_risk[col] = 35.0
    elif col == 'delinq_2yrs': high_risk[col] = 4
    elif col == 'revol_util': high_risk[col] = 92.0
    elif col == 'loan_amnt': high_risk[col] = 28000.0
    elif col == 'int_rate': high_risk[col] = 21.0
    elif col == 'fico_range_low': high_risk[col] = 580.0
    else: high_risk[col] = float(reference_defaults.get(col, 0))

for col in cat_features:
    if col == 'grade': high_risk[col] = 'E'
    elif col == 'sub_grade': high_risk[col] = 'E5'
    elif col == 'term': high_risk[col] = '60 months'
    elif col == 'home_ownership': high_risk[col] = 'RENT'
    else: high_risk[col] = reference_defaults.get(col, 'Missing')

demo_profiles[" Perfil Alto Riesgo (Mal Pagador)"] = high_risk

# ---------------------------
# PERFIL 4: TÍPICO
# ---------------------------
print(" Perfil TÍPICO...")
typical = {}
for col in num_features:
    typical[col] = float(reference_defaults.get(col, 0))
for col in cat_features:
    typical[col] = reference_defaults.get(col, 'Missing')

demo_profiles[" Perfil Típico (Valores Medios)"] = typical

# ---------------------------
# GUARDAR
# ---------------------------
joblib.dump(demo_profiles, 'demo_profiles.pkl')
print("\n demo_profiles.pkl CREADO EXITOSAMENTE")

if os.path.exists('demo_profiles.pkl'):
    size = os.path.getsize('demo_profiles.pkl') / 1024
    print(f"   Tamaño: {size:.1f} KB")
    print(f"   Perfiles: {len(demo_profiles)}")