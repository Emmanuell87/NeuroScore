import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

print("\n--- INICIANDO EVALUACIÓN DEL MODELO DE BAJA COMPLEJIDAD ---")
print("[INFO] Cargando datos y preprocesando... (esto es rápido)")

df = pd.read_csv('loan/loan.csv', low_memory=False)

cols_leakage = ['collection_recovery_fee', 'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'recoveries', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp']
cols_metadata = ['id', 'member_id', 'url', 'desc', 'title', 'emp_title', 'issue_d', 'last_credit_pull_d', 'earliest_cr_line', 'zip_code', 'policy_code']
cat_features = ['addr_state', 'application_type', 'emp_length', 'grade', 'home_ownership', 'initial_list_status', 'is_inc_v', 'purpose', 'pymnt_plan', 'sub_grade', 'term', 'verified_status_joint']
num_features = ['annual_inc', 'annual_inc_joint', 'collections_12_mths_ex_med', 'delinq_2yrs', 'dti', 'dti_joint', 'fico_range_high', 'fico_range_low', 'funded_amnt', 'funded_amnt_inv', 'inq_last_6mths', 'installment', 'int_rate', 'loan_amnt', 'last_fico_range_high', 'last_fico_range_low', 'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal']

cols_to_drop = [c for c in cols_leakage + cols_metadata if c in df.columns]
df_filtered = df.drop(columns=cols_to_drop, errors='ignore')

cat_features = [c for c in cat_features if c in df_filtered.columns]
num_features = [c for c in num_features if c in df_filtered.columns]

status_to_target = {
    'Fully Paid': 0, 'Does not meet the credit policy. Status:Fully Paid': 0,
    'Charged Off': 1, 'Default': 1, 'Does not meet the credit policy. Status:Charged Off': 1, 'Late (31-120 days)': 1,
    'Current': np.nan, 'Issued': np.nan, 'In Grace Period': np.nan, 'Late (16-30 days)': np.nan,
}

df_filtered['target'] = df_filtered['loan_status'].map(status_to_target)
df_filtered = df_filtered[df_filtered['target'].notna()].copy()
df_filtered['target'] = df_filtered['target'].astype(int)

for col in num_features:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())

for col in cat_features:
    df_filtered[col] = df_filtered[col].fillna('Missing')
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))

X = df_filtered[num_features + cat_features]
y = df_filtered['target']
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[TRAIN] Entrenando Regresión Logística (Modelo de Baja Complejidad)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

y_test_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred_label_lr = lr_model.predict(X_test_scaled)

auc_score_lr = roc_auc_score(y_test, y_test_pred_proba_lr)
brier_lr = brier_score_loss(y_test, y_test_pred_proba_lr)

print("\n" + "="*50)
print(" RESULTADOS MODELO DE BAJA COMPLEJIDAD (BASELINE)")
print("="*50)
print(f"[METRICA] AUC-ROC Score : {auc_score_lr:.4f}")
print(f"[METRICA] Brier Score   : {brier_lr:.4f}")

print("\n[REPORTE] REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_test_pred_label_lr, target_names=["Pagador (0)", "Incumplimiento (1)"]))

cm = confusion_matrix(y_test, y_test_pred_label_lr)
print("\n[MATRIZ] MATRIZ DE CONFUSIÓN:")
print(f"                       Predicción: Pagador    Predicción: Incumplimiento")
print(f"Real: Pagador          {cm[0][0]:<23} {cm[0][1]}")
print(f"Real: Incumplimiento   {cm[1][0]:<23} {cm[1][1]}")
print("="*50 + "\n")

# Guardar la matriz y el reporte en un archivo txt para el informe técnico
with open('reporte_modelo_baja_complejidad.txt', 'w') as f:
    f.write("RESUMEN DEL MODELO DE BAJA COMPLEJIDAD (Regresión Logística)\n\n")
    f.write(f"AUC: {auc_score_lr:.4f}\n")
    f.write(f"Brier: {brier_lr:.4f}\n\n")
    f.write("REPORTE DE CLASIFICACIÓN\n")
    f.write(classification_report(y_test, y_test_pred_label_lr, target_names=["Pagador (0)", "Incumplimiento (1)"]))
    f.write("\nMATRIZ DE CONFUSIÓN\n")
    f.write(f"[[{cm[0][0]}, {cm[0][1]}],\n [{cm[1][0]}, {cm[1][1]}]]\n")

print("Se ha generado el archivo 'reporte_modelo_baja_complejidad.txt' con toda esta información.")
