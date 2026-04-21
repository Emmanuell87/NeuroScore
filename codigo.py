# ==============================================
# PUNTO 1 (VERSIÓN CORREGIDA PARA TUS 85 VARIABLES)
# MODELO DE PROBABILIDAD DE INCUMPLIMIENTO CON RNA OPTIMIZADA
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import keras_tuner as kt
import warnings
import joblib

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------
# 1. CARGA DE DATOS
# ---------------------------
print("[INFO] Cargando 'loan.csv'...")
df = pd.read_csv('loan/loan.csv', low_memory=False)
print(f"Dataset original: {df.shape}")

# ---------------------------
# 2. DEFINICIÓN DE VARIABLES SEGÚN TU LISTA EXACTA
# ---------------------------

# 2.1. Variables que son FUGA DE INFORMACIÓN (Leakage) - SE ELIMINAN
cols_leakage = [
    'collection_recovery_fee', 'last_pymnt_amnt', 'last_pymnt_d', 
    'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'recoveries', 
    'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 
    'total_rec_late_fee', 'total_rec_prncp'
]

# 2.2. Variables de ID / Texto / Irrelevantes - SE ELIMINAN
cols_metadata = [
    'id', 'member_id', 'url', 'desc', 'title', 'emp_title', 
    'issue_d', 'last_credit_pull_d', 'earliest_cr_line', 
    'zip_code', 'policy_code'
]

# 2.3. Variables CATEGÓRICAS que SÍ usaremos
# (Ajusta esta lista si quieres excluir alguna. addr_state es debatible, yo la incluyo)
cat_features = [
    'addr_state', 'application_type', 'emp_length', 'grade', 
    'home_ownership', 'initial_list_status', 'is_inc_v', 'purpose', 
    'pymnt_plan', 'sub_grade', 'term', 'verified_status_joint'
]

# 2.4. Variables NUMÉRICAS que SÍ usaremos (El resto de las 85 menos las de arriba)
# Nota: Las listaremos manualmente para asegurar que no se nos cuela ninguna leakage.
num_features = [
    'annual_inc', 'annual_inc_joint', 'collections_12_mths_ex_med', 'delinq_2yrs',
    'dti', 'dti_joint', 'fico_range_high', 'fico_range_low', 'funded_amnt',
    'funded_amnt_inv', 'inq_last_6mths', 'installment', 'int_rate',
    'last_fico_range_high', 'last_fico_range_low', 'loan_amnt',
    'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
    'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
    'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
    'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal'
]

# 2.5. Asegurar que solo usamos columnas que existen en el DataFrame cargado
cols_to_drop = cols_leakage + cols_metadata
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df_filtered = df.drop(columns=cols_to_drop, errors='ignore')

cat_features = [c for c in cat_features if c in df_filtered.columns]
num_features = [c for c in num_features if c in df_filtered.columns]

print(f"[OK] Variables categóricas seleccionadas: {len(cat_features)}")
print(f"[OK] Variables numéricas seleccionadas: {len(num_features)}")

# ---------------------------
# 3. DEFINICIÓN DEL TARGET (loan_status)
# ---------------------------
# Mapeo recomendado del reto:
# 0 = bueno (pagador), 1 = malo (incumple), NA = estado sin desenlace definitivo
status_to_target = {
    'Fully Paid': 0,
    'Does not meet the credit policy. Status:Fully Paid': 0,
    'Charged Off': 1,
    'Default': 1,
    'Does not meet the credit policy. Status:Charged Off': 1,
    'Late (31-120 days)': 1,
    'Current': np.nan,
    'Issued': np.nan,
    'In Grace Period': np.nan,
    'Late (16-30 days)': np.nan,
}

df_filtered['target'] = df_filtered['loan_status'].map(status_to_target)
df_filtered = df_filtered[df_filtered['target'].notna()].copy()
df_filtered['target'] = df_filtered['target'].astype(int)

print(f"[TARGET] Tasa de Incumplimiento (Default): {df_filtered['target'].mean():.2%}")
print(f"[SHAPE] Tamaño del dataset utilizable: {df_filtered.shape}")

# ---------------------------
# 4. LIMPIEZA Y PREPROCESAMIENTO
# ---------------------------
# Variables Numéricas: Llenar NaN con Mediana y convertir a float
for col in num_features:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())

# Variables Categóricas: Llenar NaN con 'Missing' y aplicar LabelEncoder
label_encoders = {}
for col in cat_features:
    df_filtered[col] = df_filtered[col].fillna('Missing')
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
    label_encoders[col] = le

# ---------------------------
# 5. PREPARACIÓN MATRIZ X e Y
# ---------------------------
X = df_filtered[num_features + cat_features]
y = df_filtered['target']

# Eliminar filas con infinitos (por si acaso)
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

print(f"[DIMS] Dimensiones finales X: {X.shape}")

# ---------------------------
# 6. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

# Escalado (Esencial para Redes Neuronales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Entrenamiento: {X_train_scaled.shape}, Validación: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

# ---------------------------
# 7. OPTIMIZACIÓN DE RED NEURONAL CON KERAS TUNER
# ---------------------------
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
    
    # Hiperparámetros a optimizar
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu',
            kernel_regularizer=regularizers.l2(hp.Choice(f'l2_{i}', values=[1e-3, 1e-4, 1e-5]))
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')]
    )
    return model

# Inicializar Tuner
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_auc', direction='max'),
    max_epochs=20,
    factor=3,
    directory='keras_tuner_credit',
    project_name='punto1'
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n[SEARCH] Iniciando búsqueda de arquitectura óptima... (Puede tardar varios minutos)")
tuner.search(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=20,
    callbacks=[stop_early],
    batch_size=512,
    verbose=1
)

# ---------------------------
# 8. ENTRENAMIENTO FINAL Y EVALUACIÓN
# ---------------------------
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

print("\n[TRAIN] Entrenando modelo final con la mejor arquitectura...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=512,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)],
    verbose=1
)

# Evaluación
y_pred_proba = model.predict(X_test_scaled).flatten()
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n[RESULT] AUC en Test (Red Neuronal Optimizada): {auc_score:.4f}")

# Reporte de clasificación con umbral base 0.5
y_pred_label = (y_pred_proba >= 0.5).astype(int)
print("\n[REPORT] Classification report (threshold=0.50):")
print(classification_report(y_test, y_pred_label, digits=4))

# Gráfica ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'RNA (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Modelo de Incumplimiento')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# 9. SCORECARD DEL MODELO (PUNTO 2)
# ---------------------------
# Convertimos PD (probabilidad de incumplimiento) a score.
# Mayor score = menor riesgo.
PDO = 50
SCORE_AT_ODDS = 600
ODDS_REF = 20  # odds buenos:malos en punto de referencia

B = PDO / np.log(2)
A = SCORE_AT_ODDS - B * np.log(ODDS_REF)

eps = 1e-6
pd_all = model.predict(scaler.transform(X)).flatten()
pd_all = np.clip(pd_all, eps, 1 - eps)
odds_all = (1 - pd_all) / pd_all
score_all = A + B * np.log(odds_all)

scorecard_df = pd.DataFrame({
    'pd_bad': pd_all,
    'score': score_all,
    'target': y.values,
})

# Resumen tipo scorecard por deciles
scorecard_df['score_decile'] = pd.qcut(scorecard_df['score'], q=10, labels=False, duplicates='drop')
scorecard_summary = (
    scorecard_df
    .groupby('score_decile', as_index=False)
    .agg(
        n_clientes=('score', 'count'),
        score_min=('score', 'min'),
        score_max=('score', 'max'),
        score_promedio=('score', 'mean'),
        pd_promedio=('pd_bad', 'mean'),
        bad_rate_real=('target', 'mean'),
    )
    .sort_values('score_decile', ascending=False)
)

print("\n[SCORECARD] Resumen por deciles de score (1er filas):")
print(scorecard_summary.head(10).to_string(index=False))

# Posición relativa de la población para comparar en app (percentiles)
scorecard_df['percentil_poblacion'] = scorecard_df['score'].rank(pct=True) * 100

# ---------------------------
# 10. VARIABLES MAS RIESGOSAS (PUNTO 3)
# ---------------------------
# Analisis basado en importancia por permutacion (caida de AUC al desordenar cada variable)
# y direccion del riesgo (correlacion con PD predicha).
max_eval = 15000
if len(X_test) > max_eval:
    eval_idx = X_test.sample(n=max_eval, random_state=42).index
    X_eval_df = X_test.loc[eval_idx].copy()
    y_eval = y_test.loc[eval_idx].copy()
else:
    X_eval_df = X_test.copy()
    y_eval = y_test.copy()

X_eval_scaled = scaler.transform(X_eval_df)
y_eval_pred = model.predict(X_eval_scaled, verbose=0).flatten()
baseline_auc_eval = roc_auc_score(y_eval, y_eval_pred)

rng = np.random.default_rng(42)
risk_rows = []

for i, col in enumerate(X.columns):
    X_perm = X_eval_scaled.copy()
    X_perm[:, i] = rng.permutation(X_perm[:, i])
    y_perm_pred = model.predict(X_perm, verbose=0).flatten()
    auc_perm = roc_auc_score(y_eval, y_perm_pred)
    auc_drop = baseline_auc_eval - auc_perm

    col_values = X_eval_df[col].values
    if np.std(col_values) == 0:
        corr_with_pd = 0.0
    else:
        corr_with_pd = float(np.corrcoef(col_values, y_eval_pred)[0, 1])
        if np.isnan(corr_with_pd):
            corr_with_pd = 0.0

    if corr_with_pd > 0:
        risk_direction = 'A mayor valor, mayor riesgo'
    elif corr_with_pd < 0:
        risk_direction = 'A mayor valor, menor riesgo'
    else:
        risk_direction = 'Sin relacion lineal clara'

    risk_rows.append({
        'variable': col,
        'auc_drop_permutacion': auc_drop,
        'corr_con_pd': corr_with_pd,
        'direccion_riesgo': risk_direction,
    })

risk_analysis_df = pd.DataFrame(risk_rows).sort_values('auc_drop_permutacion', ascending=False)

print("\n[RISK] Top 15 variables mas influyentes en riesgo (AUC drop):")
print(risk_analysis_df.head(15).to_string(index=False))
print("\n[RISK] Nota: En variables categoricas codificadas con LabelEncoder, la direccion debe interpretarse con cautela.")

# ---------------------------
# 11. GUARDADO DE COMPONENTES (Para Scorecard y App)
# ---------------------------
joblib.dump(scaler, 'scaler_nn.pkl')
joblib.dump(label_encoders, 'label_encoders_nn.pkl')
joblib.dump(list(X.columns), 'feature_names_nn.pkl')
joblib.dump({'A': A, 'B': B, 'PDO': PDO, 'score_at_odds': SCORE_AT_ODDS, 'odds_ref': ODDS_REF}, 'scorecard_params.pkl')

scorecard_df.to_csv('scorecard_poblacion.csv', index=False)
scorecard_summary.to_csv('scorecard_resumen_deciles.csv', index=False)
risk_analysis_df.to_csv('analisis_variables_riesgo.csv', index=False)

model.save('modelo_nn_credit_risk.h5')

print("[SAVED] Componentes guardados: modelo, scaler, encoders, features, scorecard y analisis de riesgo")