import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf
import altair as alt


st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")

# Reemplaza estos enlaces con tus URLs finales de entrega
TECH_REPORT_URL = "https://example.com/reporte-tecnico"
MARKETING_MATERIAL_URL = "https://example.com/material-publicitario"

TARGET_MAPPING = [
    {
        "loan_status": "Current",
        "target": "NA",
        "justificacion": "Sin desenlace final de pago confirmado.",
    },
    {
        "loan_status": "Fully Paid",
        "target": 0,
        "justificacion": "Buen pagador confirmado.",
    },
    {
        "loan_status": "Charged Off",
        "target": 1,
        "justificacion": "Incumplimiento confirmado.",
    },
    {
        "loan_status": "Late (31-120 days)",
        "target": 1,
        "justificacion": "Se considera mal pagador.",
    },
    {
        "loan_status": "Issued",
        "target": "NA",
        "justificacion": "Sin historial de pago suficiente.",
    },
    {
        "loan_status": "In Grace Period",
        "target": "NA",
        "justificacion": "Aun no hay incumplimiento confirmado.",
    },
    {
        "loan_status": "Late (16-30 days)",
        "target": "NA",
        "justificacion": "Comportamiento final aun incierto.",
    },
    {
        "loan_status": "Does not meet the credit policy. Status:Fully Paid",
        "target": 0,
        "justificacion": "Buen pagador confirmado.",
    },
    {
        "loan_status": "Default",
        "target": 1,
        "justificacion": "Impago confirmado.",
    },
    {
        "loan_status": "Does not meet the credit policy. Status:Charged Off",
        "target": 1,
        "justificacion": "Incumplimiento confirmado.",
    },
]

LOAN_STATUS_TRANSLATIONS = {
    "Current": "Al dia (Current)",
    "Fully Paid": "Pagado completamente (Fully Paid)",
    "Charged Off": "Incobrable / Castigado (Charged Off)",
    "Late (31-120 days)": "Mora 31-120 días",
    "Issued": "Crédito emitido sin historial (Issued)",
    "In Grace Period": "Periodo de gracia",
    "Late (16-30 days)": "Mora 16-30 días",
    "Does not meet the credit policy. Status:Fully Paid": "No cumple política, pero pagó",
    "Default": "Incumplimiento confirmado (Default)",
    "Does not meet the credit policy. Status:Charged Off": "No cumple política e incobrable",
}

FRIENDLY_LABELS = {
    "annual_inc": "Ingreso anual",
    "annual_inc_joint": "Ingreso anual conjunto",
    "collections_12_mths_ex_med": "Cobros en 12 meses (sin medicos)",
    "delinq_2yrs": "Delincuencias en 2 años",
    "dti": "Relacion deuda/ingreso (DTI)",
    "dti_joint": "DTI conjunto",
    "funded_amnt": "Monto financiado",
    "funded_amnt_inv": "Monto financiado por inversionistas",
    "inq_last_6mths": "Consultas de crédito últimos 6 meses",
    "installment": "Cuota mensual",
    "int_rate": "Tasa de interes",
    "loan_amnt": "Monto del prestamo",
    "mths_since_last_delinq": "Meses desde última mora",
    "mths_since_last_major_derog": "Meses desde ultimo deterioro mayor",
    "mths_since_last_record": "Meses desde ultimo registro",
    "open_acc": "Cuentas abiertas",
    "pub_rec": "Registros publicos negativos",
    "revol_bal": "Saldo revolvente",
    "revol_util": "Utilizacion revolvente",
    "total_acc": "Total de cuentas",
    "open_acc_6m": "Cuentas abiertas en 6 meses",
    "open_il_6m": "Cuentas a plazos abiertas en 6 meses",
    "open_il_12m": "Cuentas a plazos abiertas en 12 meses",
    "open_il_24m": "Cuentas a plazos abiertas en 24 meses",
    "mths_since_rcnt_il": "Meses desde ultima cuenta a plazos",
    "total_bal_il": "Saldo total en cuentas a plazos",
    "il_util": "Utilizacion de cuentas a plazos",
    "open_rv_12m": "Revolving abiertas en 12 meses",
    "open_rv_24m": "Revolving abiertas en 24 meses",
    "max_bal_bc": "Maximo saldo en tarjeta bancaria",
    "all_util": "Utilizacion total",
    "total_rev_hi_lim": "Limite total revolvente",
    "inq_fi": "Consultas financieras",
    "total_cu_tl": "Total líneas de crédito actuales",
    "inq_last_12m": "Consultas ultimos 12 meses",
    "acc_now_delinq": "Cuentas actualmente en mora",
    "tot_coll_amt": "Monto total en cobranza",
    "tot_cur_bal": "Saldo total actual",
    "addr_state": "Estado",
    "application_type": "Tipo de aplicacion",
    "emp_length": "Antiguedad laboral",
    "grade": "Grado de crédito",
    "home_ownership": "Tenencia de vivienda",
    "initial_list_status": "Estado inicial",
    "purpose": "Proposito del prestamo",
    "pymnt_plan": "Plan de pago",
    "sub_grade": "Subgrado de crédito",
    "term": "Plazo",
}

VALUE_TRANSLATIONS = {
    "home_ownership": {
        "MORTGAGE": "Hipoteca",
        "RENT": "Arriendo",
        "OWN": "Propia",
        "ANY": "Cualquiera",
        "NONE": "Ninguna",
        "OTHER": "Otra",
    },
    "purpose": {
        "debt_consolidation": "Consolidacion de deudas",
        "credit_card": "Tarjeta de crédito",
        "home_improvement": "Mejora de vivienda",
        "major_purchase": "Compra mayor",
        "small_business": "Negocio pequeno",
        "car": "Auto",
        "medical": "Gastos medicos",
        "vacation": "Vacaciones",
        "moving": "Mudanza",
        "house": "Vivienda",
        "wedding": "Boda",
        "renewable_energy": "Energia renovable",
        "educational": "Educacion",
        "other": "Otro",
    },
    "term": {
        "36 months": "36 meses",
        "60 months": "60 meses",
    },
    "application_type": {
        "INDIVIDUAL": "Individual",
        "JOINT": "Conjunta",
    },
    "pymnt_plan": {
        "n": "No",
        "y": "Si",
    },
    "initial_list_status": {
        "f": "Fraccional",
        "w": "Completo",
    },
}

INTEGER_FEATURES = {
    "collections_12_mths_ex_med",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_major_derog",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "total_acc",
    "open_acc_6m",
    "open_il_6m",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "open_rv_12m",
    "open_rv_24m",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_now_delinq",
}


def pretty_label(col_name):
    return FRIENDLY_LABELS.get(col_name, col_name)


def pretty_category_value(col_name, raw_value):
    raw_text = str(raw_value)
    mapped = VALUE_TRANSLATIONS.get(col_name, {}).get(raw_text)
    if mapped is not None:
        return mapped

    # Fallback legible si no hay traduccion manual para ese valor.
    return raw_text.replace("_", " ").replace("-", " ").title()


def pretty_loan_status(raw_status):
    text = str(raw_status)
    return LOAN_STATUS_TRANSLATIONS.get(text, text)


def build_full_demo_profiles(feature_names, cat_features):
    loan = pd.read_csv("loan/loan.csv", low_memory=False)

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

    profiles = {
        "Caso real - Bajo riesgo": row_to_profile(loan.loc[low_idx]),
        "Caso real - Riesgo medio": row_to_profile(loan.loc[med_idx]),
        "Caso real - Alto riesgo": row_to_profile(loan.loc[high_idx]),
    }

    return profiles


def build_widget_profile(demo_choice, mode, demo_cases, defaults, core_numeric, core_categorical, numeric_features, cat_features):
    if mode == "Basico (normal, recomendado)":
        numeric_targets = core_numeric
        categorical_targets = core_categorical
        numeric_prefix = "basic_num"
        categorical_prefix = "basic_cat"
    else:
        numeric_targets = numeric_features
        categorical_targets = cat_features
        numeric_prefix = "adv_num"
        categorical_prefix = "adv_cat"

    profile = {}
    selected_demo = demo_cases.get(demo_choice, {}) if demo_choice != "Ninguno" else {}

    for col in numeric_targets:
        if demo_choice == "Ninguno":
            profile[f"{numeric_prefix}_{col}"] = 0
        else:
            profile[f"{numeric_prefix}_{col}"] = selected_demo.get(col, defaults[col])

    for col in categorical_targets:
        if demo_choice == "Ninguno":
            profile[f"{categorical_prefix}_{col}"] = defaults[col]
        else:
            profile[f"{categorical_prefix}_{col}"] = selected_demo.get(col, defaults[col])

    return profile


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("modelo_nn_credit_risk.h5", compile=False)
    scaler = joblib.load("scaler_nn.pkl")
    label_encoders = joblib.load("label_encoders_nn.pkl")
    feature_names = joblib.load("feature_names_nn.pkl")
    score_params = joblib.load("scorecard_params.pkl")
    return model, scaler, label_encoders, feature_names, score_params


@st.cache_data
def load_population_data():
    score_pop = pd.read_csv("scorecard_poblacion.csv")
    score_deciles = pd.read_csv("scorecard_resumen_deciles.csv")
    risk_analysis = pd.read_csv("analisis_variables_riesgo.csv")
    return score_pop, score_deciles, risk_analysis


@st.cache_data
def build_reference_values(feature_names, cat_features):
    df = pd.read_csv("loan/loan.csv", low_memory=False)
    defaults = {}

    for col in feature_names:
        if col in cat_features:
            if col in df.columns:
                mode_series = df[col].dropna().astype(str)
                defaults[col] = mode_series.mode().iloc[0] if not mode_series.empty else "Missing"
            else:
                defaults[col] = "Missing"
        else:
            if col in df.columns:
                num = pd.to_numeric(df[col], errors="coerce")
                med = num.median()
                defaults[col] = float(med) if pd.notna(med) else 0.0
            else:
                defaults[col] = 0.0

    return defaults


def compute_score(pd_bad, score_params):
    pd_bad = np.clip(pd_bad, 1e-6, 1 - 1e-6)
    odds = (1 - pd_bad) / pd_bad
    a = score_params["A"]
    b = score_params["B"]
    return float(a + b * np.log(odds))


def encode_input(user_input, feature_names, label_encoders):
    row = {}

    for col in feature_names:
        if col in label_encoders:
            encoder = label_encoders[col]
            classes = encoder.classes_.tolist()
            value = str(user_input[col])

            if value not in classes:
                value = "Missing" if "Missing" in classes else classes[0]

            row[col] = int(encoder.transform([value])[0])
        else:
            row[col] = float(user_input[col])

    return pd.DataFrame([row], columns=feature_names)


def render_numeric_inputs(feature_list, defaults, values_store, key_prefix, n_cols=3):
    cols = st.columns(n_cols)
    for idx, col in enumerate(feature_list):
        with cols[idx % n_cols]:
            raw_value = values_store.get(col, defaults[col])

            if col in INTEGER_FEATURES:
                values_store[col] = int(
                    st.number_input(
                        label=pretty_label(col),
                        value=int(round(float(raw_value))),
                        step=1,
                        format="%d",
                        key=f"{key_prefix}_{col}",
                    )
                )
            else:
                values_store[col] = st.number_input(
                    label=pretty_label(col),
                    value=float(raw_value),
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_{col}",
                )


def render_categorical_inputs(feature_list, defaults, values_store, label_encoders, key_prefix, n_cols=3):
    cols = st.columns(n_cols)
    for idx, col in enumerate(feature_list):
        classes = label_encoders[col].classes_.tolist()
        default_val = str(values_store.get(col, defaults[col]))
        default_idx = classes.index(default_val) if default_val in classes else 0

        with cols[idx % n_cols]:
            values_store[col] = st.selectbox(
                label=pretty_label(col),
                options=classes,
                index=default_idx,
                format_func=lambda x, c=col: pretty_category_value(c, x),
                key=f"{key_prefix}_{col}",
            )


def main():
    st.title("Credit Risk Scorecard")
    st.write("Aplicación para estimar probabilidad de incumplimiento, score crediticio y comparación contra la población.")

    st.sidebar.header("Entregables")
    st.sidebar.markdown(f"[Reporte técnico]({TECH_REPORT_URL})")
    st.sidebar.markdown(f"[Material publicitario]({MARKETING_MATERIAL_URL})")
    

    model, scaler, label_encoders, feature_names, score_params = load_artifacts()
    score_pop, score_deciles, risk_analysis = load_population_data()

    score_deciles_view = score_deciles.rename(
        columns={
            "score_decile": "Decil",
            "n_clientes": "Numero de clientes",
            "score_min": "Score minimo",
            "score_max": "Score maximo",
            "score_promedio": "Score promedio",
            "pd_promedio": "PD promedio",
            "bad_rate_real": "Tasa real de incumplimiento",
        }
    )

    risk_analysis_view = risk_analysis.rename(
        columns={
            "variable": "Variable",
            "auc_drop_permutacion": "Importancia (caida AUC)",
            "corr_con_pd": "Correlacion con riesgo",
            "direccion_riesgo": "Interpretacion",
        }
    )
    risk_analysis_view["Variable"] = risk_analysis_view["Variable"].map(pretty_label)

    target_mapping_view = pd.DataFrame(TARGET_MAPPING).rename(
        columns={
            "loan_status": "Estado original",
            "target": "Target binario",
            "justificacion": "Justificacion",
        }
    )
    target_mapping_view["Estado original"] = target_mapping_view["Estado original"].map(pretty_loan_status)
    target_mapping_view["Target binario"] = target_mapping_view["Target binario"].replace({"NA": "No aplica (se excluye)"})

    cat_features = list(label_encoders.keys())
    numeric_features = [c for c in feature_names if c not in cat_features]
    defaults = build_reference_values(feature_names, cat_features)
    widget_defaults = defaults.copy()
    for col in numeric_features:
        widget_defaults[col] = 0

    demo_cases = build_full_demo_profiles(feature_names, cat_features)

    core_numeric = [
        c for c in [
            "annual_inc", "loan_amnt", "int_rate", "dti", "installment",
            "open_acc", "revol_util", "total_acc", "inq_last_6mths",
        ] if c in numeric_features
    ]
    core_categorical = [
        c for c in ["grade", "sub_grade", "term", "home_ownership", "purpose", "emp_length"] if c in cat_features
    ]

    tab_pred, tab_target, tab_model = st.tabs([
        "Predicción", "Definición Target (0/1/NA)", "Analítica del Modelo"
    ])

    with tab_pred:
        st.subheader("Datos de entrada")
        st.caption("Modo Básico: completa solo variables clave. El resto usa valores típicos de la población.")

        demo_choice = st.selectbox(
            "Casos de prueba",
            ["Ninguno"] + list(demo_cases.keys()),
            index=0,
            key="demo_choice",
        )
        if st.button("Cargar caso de prueba"):
            mode_selected = st.session_state.get("capture_mode", "Basico (normal, recomendado)")
            widget_profile = build_widget_profile(
                st.session_state.demo_choice,
                mode_selected,
                demo_cases,
                defaults,
                core_numeric,
                core_categorical,
                numeric_features,
                cat_features,
            )
            st.session_state.demo_profile = demo_cases.get(st.session_state.demo_choice, {})
            st.session_state.demo_applied_mode = mode_selected
            for key, value in widget_profile.items():
                st.session_state[key] = value
            st.rerun()

        if st.button("Limpiar caso de prueba"):
            st.session_state.demo_profile = {}
            st.session_state.demo_applied_mode = None
            for col in numeric_features:
                st.session_state.pop(f"adv_num_{col}", None)
            for col in core_numeric:
                st.session_state.pop(f"basic_num_{col}", None)
                st.session_state.pop(f"adv_num_{col}", None)
            for col in core_categorical:
                st.session_state.pop(f"basic_cat_{col}", None)
                st.session_state.pop(f"adv_cat_{col}", None)
            for col in cat_features:
                st.session_state.pop(f"adv_cat_{col}", None)
            st.rerun()

        mode = st.radio(
            "Modo de captura",
            ["Basico (normal, recomendado)", "Avanzado"],
            horizontal=True,
            key="capture_mode",
        )
        st.caption(
            "Básico (normal): pides solo variables clave y el resto toma valores típicos. "
            "Avanzado: pides todas las variables del modelo."
        )

        if st.session_state.get("demo_profile"):
            if st.session_state.get("demo_applied_mode") != mode:
                active_demo_choice = st.session_state.get("demo_choice", "Ninguno")
                widget_profile = build_widget_profile(
                    active_demo_choice,
                    mode,
                    demo_cases,
                    defaults,
                    core_numeric,
                    core_categorical,
                    numeric_features,
                    cat_features,
                )
                for key, value in widget_profile.items():
                    st.session_state[key] = value
                st.session_state.demo_applied_mode = mode
                st.rerun()

        with st.form("prediction_form"):
            user_input = widget_defaults.copy()
            if "demo_profile" in st.session_state and st.session_state.demo_profile:
                user_input.update(st.session_state.demo_profile)

            if mode == "Basico (normal, recomendado)":
                st.info("En este modo solo editas variables clave. Las demás variables se completan automáticamente con valores de referencia.")
                st.markdown("### Variables numéricas clave")
                render_numeric_inputs(core_numeric, widget_defaults, user_input, key_prefix="basic_num", n_cols=3)

                st.markdown("### Variables categóricas clave")
                render_categorical_inputs(core_categorical, widget_defaults, user_input, label_encoders, key_prefix="basic_cat", n_cols=3)
            else:
                st.info("Modo avanzado activo: puedes editar todas las variables que usa el modelo.")
                st.markdown("### Variables numéricas")
                render_numeric_inputs(numeric_features, widget_defaults, user_input, key_prefix="adv_num", n_cols=3)

                st.markdown("### Variables categóricas")
                render_categorical_inputs(cat_features, widget_defaults, user_input, label_encoders, key_prefix="adv_cat", n_cols=3)

            submitted = st.form_submit_button("Calcular score")

        if submitted:
            x_input = encode_input(user_input, feature_names, label_encoders)
            x_scaled = scaler.transform(x_input)
            pd_bad = float(model.predict(x_scaled, verbose=0).flatten()[0])
            score = compute_score(pd_bad, score_params)

            score_percentile = float((score_pop["score"] <= score).mean() * 100)
            pd_percentile = float((score_pop["pd_bad"] <= pd_bad).mean() * 100)

            st.subheader("Resultado")
            c1, c2, c3 = st.columns(3)
            c1.metric("PD (prob. incumplimiento)", f"{pd_bad:.2%}")
            c2.metric("Score", f"{score:.1f}")
            c3.metric("Percentil de score", f"{score_percentile:.1f}")

            if pd_bad >= 0.30:
                band = "Alto riesgo"
            elif pd_bad >= 0.15:
                band = "Riesgo medio"
            else:
                band = "Bajo riesgo"

            st.info(f"Clasificación estimada: {band}")
            st.caption(f"Percentil por PD (más alto = más riesgoso): {pd_percentile:.1f}")

            st.markdown("### Tu posición vs población (distribución de score)")
            hist = alt.Chart(score_pop).mark_bar(opacity=0.55).encode(
                x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Score"),
                y=alt.Y("count():Q", title="Frecuencia"),
            )
            mark_user = alt.Chart(pd.DataFrame({"score": [score]})).mark_rule(color="red", size=3).encode(
                x="score:Q"
            )
            st.altair_chart((hist + mark_user).properties(height=320), use_container_width=True)

        st.subheader("Comparación por deciles")
        st.caption(
            "Los deciles dividen la población en 10 grupos de tamaño similar según su score. "
            "Sirven para comparar rápidamente el riesgo relativo entre segmentos."
        )
        st.dataframe(score_deciles_view, use_container_width=True)

    with tab_target:
        st.subheader("Reglas de construcción de la variable objetivo")
        st.write("El modelo predice probabilidad de ser cliente malo (target = 1).")
        st.dataframe(target_mapping_view, use_container_width=True)
        st.warning("Los casos con target = NA se excluyen del entrenamiento para evitar ruido en la etiqueta.")

    with tab_model:
        st.subheader("Variables más riesgosas del modelo")
        st.caption("Ordenadas por caída de AUC al permutar la variable (mayor caída = mayor relevancia).")
        st.dataframe(risk_analysis_view.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
