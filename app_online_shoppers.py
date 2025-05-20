import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Seiteneinstellungen
st.set_page_config(
    page_title="🛍️ Kaufvorhersage basierend auf Browsing-Verhalten",
    page_icon="🚀",
    layout="wide"
)

st.title("🛍️ Kaufvorhersage basierend auf Browsing-Verhalten")
tab1, tab2 = st.tabs(["Kaufvorhersage", "Variablen Übersicht"])

# Tab 1: Vorhersage
with tab1:
    st.markdown("Gib die Daten einer Besuchssession ein, um Vorhersagen aus verschiedenen trainierten Modellen zu sehen.")

    @st.cache_data
    def load_data():
        df = pd.read_csv("online_shoppers_intention.csv")
        df['Revenue'] = df['Revenue'].astype(int)
        return df

    df = load_data()
    mean_vals = df.select_dtypes(include=np.number).mean()

    variable_info = pd.DataFrame({
        "Variable": [
            "Informational", "BounceRates", "ExitRates",
            "PageValues", "SpecialDay", "Month", "OperatingSystems", 
            "VisitorType", "Weekend"
        ],
        "Description": [
            "Count of informational pages visited",
            "Proportion of visitors who left the site after viewing a single page",
            "Proportion of visitors who left the site during the session",
            "Value of a page based on its contribution to revenue",
            "Proximity to a significant shopping day (e.g., Mother's Day)",
            "Month of the session",
            "Operating system used by the visitor",
            "Type of visitor (e.g., Returning Visitor, New Visitor)",
            "Whether the session occurred during the weekend",
        ]
    })

    with st.form("input_form"):
        st.subheader("🔧 Besuchsdaten eingeben (automatisch generiert)")
        input_dict = {}

        col1, col2 = st.columns(2)

        for i, row in variable_info.iterrows():
            var = row["Variable"]
            desc = row["Description"]
            label = f"{var}: {desc}"

            col = col1 if i % 2 == 0 else col2

            if var in df.select_dtypes(include='number').columns:
                min_val = 0.0
                max_val = float(df[var].max())
                default = float(round(mean_vals[var], 2))
                if df[var].dtype == 'int64' and df[var].nunique() < 30:
                    value = col.slider(label, int(min_val), int(max_val), int(default))
                else:
                    value = col.number_input(label, value=default)
            elif var in df.columns:
                options = sorted(df[var].dropna().unique())
                value = col.selectbox(label, options)
            else:
                value = col.text_input(label)

            input_dict[var] = value

        submitted = st.form_submit_button("📈 Vorhersagen anzeigen")

    if submitted:
        input_df = pd.DataFrame([input_dict])

        # Sicherstellen, dass Weekend und OperatingSystems korrekt sind
        input_df["Weekend"] = input_df["Weekend"].astype(int)
        input_df["OperatingSystems"] = pd.to_numeric(input_df["OperatingSystems"])
        df["Weekend"] = df["Weekend"].astype(int)
        df["OperatingSystems"] = pd.to_numeric(df["OperatingSystems"])

        selected_features = [
            "Informational", "BounceRates", "ExitRates", "PageValues", "SpecialDay",
            "Month", "OperatingSystems", "VisitorType", "Weekend"
        ]
        df = df[selected_features + ["Revenue"]]

        categorical_cols = ["Month", "VisitorType", "Weekend"]
        full_df = pd.concat([df.drop("Revenue", axis=1), input_df], ignore_index=True)
        full_df["Weekend"] = full_df["Weekend"].astype(int)  # Nochmals sicherstellen

        # One-Hot-Encoding
        full_encoded = pd.get_dummies(full_df, columns=categorical_cols, drop_first=True)
        input_encoded = full_encoded.tail(1)

        # Referenz-Feature-Set
        model_reference_X = pd.get_dummies(df.drop("Revenue", axis=1), columns=categorical_cols, drop_first=True)
        missing_cols = set(model_reference_X.columns) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
        input_encoded = input_encoded[model_reference_X.columns]

        # Modelle laden
        models = {
            "Baseline": joblib.load("baseline_model.pkl"),
            "Logistic Regression": joblib.load("logreg_model.pkl"),
            "Decision Tree": joblib.load("tree_model.pkl"),
            "Random Forest": joblib.load("best_rf_model.pkl"),
            "XGB": joblib.load("xgb_model.pkl"),
            "LightLGB": joblib.load("lgbm_model.pkl"),
            "Final Stacking Model": joblib.load("stacking_model.pkl"),
        }

        st.subheader("📊 Modellvorhersagen")
        for name, model in models.items():
            try:
                pred = model.predict(input_encoded)[0]
                proba = model.predict_proba(input_encoded)[0][1] if hasattr(model, "predict_proba") else None
                st.write(
                    f"**{name}** sagt: **{'🟢 Kauf' if pred == 1 else '🔴 Kein Kauf'}**",
                    f"(Wahrscheinlichkeit: {round(proba, 2)})" if proba is not None else ""
                )
            except Exception as e:
                st.error(f"{name} konnte keine Vorhersage treffen: {e}")

# Tab 2: Variablenübersicht
with tab2:
    st.subheader("📘 Variablenbeschreibung")
    st.dataframe(variable_info)
