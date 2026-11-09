from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "notebook_model.joblib"

st.set_page_config(page_title="Parkinson's Speech Classifier", page_icon="🎙️", layout="wide")


@st.cache_resource
def load_model_bundle():
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("The saved file does not contain a model bundle.")
    input_features = list(bundle.get("input_features", []))
    speech_features = list(bundle.get("feature_names", []))
    if len(input_features) != 82 or len(speech_features) != 80:
        raise ValueError("Expected 80 speech features plus Age and Sex (82 inputs).")
    if input_features[-2:] != ["Age", "Sex"]:
        raise ValueError("The saved model input order is not the expected order.")
    if int(bundle["model"].n_features_in_) != len(input_features):
        raise ValueError("The model's input count does not match the saved feature list.")
    return bundle, input_features, speech_features


def display_feature_name(name: str, speech_features: list[str]) -> str:
    # Training columns use _x for the mean-feature sheets and _y for variance sheets.
    if name.endswith("_x"):
        return name[:-2] + "_mean"
    if name.endswith("_y"):
        return name[:-2] + "_var"
    return name


def normalize_uploaded_columns(table: pd.DataFrame, input_features: list[str], speech_features: list[str]) -> pd.DataFrame:
    """Accept user-facing *_mean/*_var columns and map them to the model's *_x/*_y names."""
    rename = {}
    for model_name in speech_features:
        friendly_name = display_feature_name(model_name, speech_features)
        if friendly_name in table.columns and model_name not in table.columns:
            rename[friendly_name] = model_name
    return table.rename(columns=rename)


def validate_row(values: dict, input_features: list[str]) -> pd.DataFrame:
    missing = [name for name in input_features if name not in values or values[name] is None or str(values[name]).strip() == ""]
    if missing:
        raise ValueError(f"Missing values: {', '.join(missing[:8])}")
    row = pd.DataFrame([[values[name] for name in input_features]], columns=input_features)
    row = row.apply(pd.to_numeric, errors="coerce")
    if row.isna().any().any() or not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("Every feature, Age, and Sex must have a finite numeric value.")
    if row["Age"].iloc[0] <= 0 or row["Age"].iloc[0] > 120:
        raise ValueError("Enter a valid age between 1 and 120.")
    if row["Sex"].iloc[0] not in (0, 1):
        raise ValueError("Sex must be 0 (male) or 1 (female).")
    return row


def show_prediction(bundle: dict, row: pd.DataFrame):
    model = bundle["model"]
    probability = float(model.predict_proba(row)[0, list(model.classes_).index(1)])
    predicted_class = int(model.predict(row)[0])
    if predicted_class == 1:
        st.warning("Model prediction: Parkinson's disease (PwPD) class")
    else:
        st.success("Model prediction: Healthy control (HC) class")
    st.metric("Model-estimated PwPD probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "Experimental research output only. This model prediction is not a medical diagnosis "
        "and must not be used to make treatment decisions."
    )


st.title("Parkinson's Speech Feature Classifier")
st.write("Enter 80 precomputed speech features plus Age and Sex, or upload one person's 82-column CSV/XLSX row.")
st.info(
    "This app uses your notebook's saved **Speech + Age + Sex** model. It does not process raw audio; "
    "speech features must be calculated using the same method as the training data."
)

try:
    bundle, input_features, speech_features = load_model_bundle()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

with st.expander("About the model"):
    st.write(f"**Model:** {bundle.get('model_name', 'Saved notebook model')}")
    st.write("**Inputs:** 80 speech features + Age + Sex (82 total)")
    st.write("**Classifier:** k-nearest neighbors (KNN)")
    st.caption("Dataset is small; performance estimates are exploratory and require independent validation.")

# CSV template uses the exact training column names so column order can be restored safely.
template_columns = [display_feature_name(name, speech_features) if name not in ("Age", "Sex") else name for name in input_features]
template = pd.DataFrame([{name: "" for name in template_columns}])
st.download_button(
    "Download 82-feature input template (CSV)",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="parkinson_speech_age_sex_template.csv",
    mime="text/csv",
)

manual_tab, file_tab = st.tabs(["Enter values", "Upload a file"])
with manual_tab:
    st.subheader("Speech features")
    with st.form("feature_form"):
        form_values = {}
        for family in ["LPC", "LAR", "Cep", "MFCC"]:
            family_features = [n for n in speech_features if n.startswith(family)]
            with st.expander(f"{family} features", expanded=(family == "LPC")):
                col_a, col_b = st.columns(2)
                for idx, name in enumerate(family_features):
                    friendly_name = display_feature_name(name, speech_features)
                    with (col_a if idx % 2 == 0 else col_b):
                        form_values[name] = st.number_input(
                            friendly_name, value=None, step=0.000001, format="%.8f",
                            key=f"manual_{name}", placeholder="Enter numeric value"
                        )
        st.subheader("Demographic features")
        col1, col2 = st.columns(2)
        with col1:
            form_values["Age"] = st.number_input("Age (years)", min_value=1, max_value=120, value=None, step=1, key="manual_Age")
        with col2:
            sex_choice = st.selectbox("Sex", ["Male", "Female"], index=None, placeholder="Choose sex")
            form_values["Sex"] = None if sex_choice is None else (0 if sex_choice == "Male" else 1)
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        try:
            row = validate_row(form_values, input_features)
            show_prediction(bundle, row)
        except Exception as exc:
            st.error(str(exc))

with file_tab:
    st.subheader("Upload one person's feature row")
    uploaded = st.file_uploader("Upload CSV or Excel (.xlsx)", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            table = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            table = normalize_uploaded_columns(table, input_features, speech_features)
            missing = [name for name in input_features if name not in table.columns]
            if missing:
                st.error("File is missing required columns: " + ", ".join(missing[:8]))
            elif len(table) != 1:
                st.error("Upload exactly one row: one person's 80 speech features, Age, and Sex.")
            else:
                row = validate_row(table.iloc[0].to_dict(), input_features)
                st.success("All 82 required input columns found.")
                with st.expander("Review input values"):
                    review = row.T.rename(columns={0: "Value"})
                    review.index = [display_feature_name(n, speech_features) if n not in ("Age", "Sex") else n for n in review.index]
                    st.dataframe(review, use_container_width=True)
                if st.button("Predict from uploaded row", type="primary"):
                    show_prediction(bundle, row)
        except Exception as exc:
            st.error(f"Could not read or validate this file: {exc}")

st.divider()
st.caption("Research/classroom prototype only. Not for diagnosis or treatment decisions.")
