# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from typing import Optional, List

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="NBA Playoff Blueprint", layout="wide", page_icon="🏀")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nba_playoff_forest.joblib")
DATA_PATH = os.path.join(BASE_DIR, "nba_modeling_data_silver.csv")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please place 'nba_playoff_forest.joblib' in the app directory.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        # not fatal; data is optional for the simulator but useful for examples
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.warning(f"Could not read data file: {e}")
        return None

def get_model_feature_names(model) -> Optional[List[str]]:
    """
    Attempt to discover the feature names the model expects.
    Returns list of feature names or None if not found.
    """
    # 1) sklearn estimators often store feature_names_in_
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # 2) If model is a Pipeline, try to get final estimator's feature_names_in_
    try:
        # common pipeline attribute: named_steps, then access final estimator
        if hasattr(model, "named_steps"):
            final = list(model.named_steps.values())[-1]
            if hasattr(final, "feature_names_in_"):
                return list(final.feature_names_in_)
    except Exception:
        pass

    # 3) If model has attribute 'feature_names' (custom), try it
    try:
        if hasattr(model, "feature_names"):
            return list(model.feature_names)
    except Exception:
        pass

    # Unknown
    return None

def align_input_df(input_df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    """
    Reorder/adjust input_df to match expected list.
    If columns missing, fill with zeros and warn.
    If extra columns present, drop them.
    """
    # Find intersection in the expected order
    cols_present = [c for c in expected if c in input_df.columns]
    missing = [c for c in expected if c not in input_df.columns]
    if missing:
        st.warning(f"Model expects columns not present in inputs. Missing: {missing}. Filling missing with 0.")
        for m in missing:
            input_df[m] = 0
    # Now ensure columns are in expected order
    input_df = input_df[expected]
    return input_df

# ---------------------------------------------------------
# FEATURE DEFINITIONS (these are the sliders shown to the user)
# IMPORTANT: If your model was trained with DIFFERENT column names, update the 'key' values below
# to match the column names used during training, or set model_expected_cols later below.
# ---------------------------------------------------------
FEATURE_DEFS = [
    # (column_key, label, default)
    ("off_eFG_rank", "Offensive eFG% Rank (1 = best)", 15),
    ("def_eFG_rank", "Defensive eFG% Rank (1 = best)", 15),
    ("ft_rate_rank", "Free Throw Rate Rank", 15),
    ("opp_ft_rate_rank", "Opp. Free Throw Rate Rank", 15),
    ("tov_percent_rank", "Turnover % Rank (1 = fewest)", 15),
    ("forced_tov_rank", "Forced Turnover Rank", 15),
    ("off_reb_rank", "Offensive Rebound % Rank", 15),
    ("def_reb_rank", "Defensive Rebound % Rank", 15),
    ("ast_to_tov_rank", "Assist-to-Turnover Rank", 15),
    ("ast_ratio_rank", "Assist Ratio Rank", 15),
    ("est_pace_rank", "Estimated Pace Rank", 15),
    ("pace_per40_rank", "Pace Per 40 Rank", 15),
    ("possessions_rank", "Possessions Rank", 15),
]

# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
def main():
    st.title("🏀 NBA Playoff Blueprint: The Rank Model")
    st.markdown(
        """
        **Project by Brock Olson & Bryce Myers**

        Adjust the sliders to set your team's League Rank (1st = best, 30th = worst).
        The model predicts the probability that a team with that statistical profile makes the playoffs.
        """
    )

    model = load_model()
    df = load_data()

    if model is None:
        st.info("No model loaded. Place the model file and reload the app.")
        return

    # discover model feature names if possible
    model_expected = get_model_feature_names(model)
    if model_expected:
        st.sidebar.success("Model feature names discovered automatically.")
    else:
        st.sidebar.warning(
            "Could not auto-detect model feature names. The app will use default slider keys. "
            "If predictions fail or are wrong, edit FEATURE_DEFS keys to match training column names or set 'model_expected_cols' below."
        )

    # Sidebar: navigation and optional sample selection
    page = st.sidebar.radio("Navigate", ["GM Simulator", "Model Insights"])
    if page == "GM Simulator":
        render_simulator(model, df, model_expected)
    else:
        render_insights(df)

# ---------------------------------------------------------
# SIMULATOR
# ---------------------------------------------------------
def render_simulator(model, df, model_expected):
    st.header("🛠️ General Manager Simulator")
    st.info("Set each rank between 1 (best) and 30 (worst).")

    # Input collection
    input_data = {}
    # Create three columns for layout
    col1, col2, col3 = st.columns(3)

    # Split FEATURE_DEFS into roughly equal groups for each column
    groups = [
        FEATURE_DEFS[0:4],   # col1
        FEATURE_DEFS[4:8],   # col2
        FEATURE_DEFS[8:13],  # col3
    ]

    for col, group in zip((col1, col2, col3), groups):
        with col:
            for key, label, default in group:
                input_data[key] = st.slider(label, 1, 30, default, help=label)

    input_df = pd.DataFrame([input_data])

    # If model_expected provided, align columns
    if model_expected:
        try:
            # If the model expects more columns than the simple slider keys (for example if you used encoders),
            # align_input_df will fill missing ones with 0. This is safer than failing at predict time.
            input_df = align_input_df(input_df, model_expected)
        except Exception as e:
            st.warning(f"Could not align inputs to model feature names automatically: {e}")

    st.markdown("---")
    if st.button("Run Simulation", type="primary"):
        # Prediction
        try:
            proba = model.predict_proba(input_df)
            # proba shape is (n_rows, n_classes). We take class 1 probability for the first row.
            playoff_prob = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
            st.write(
                "Common cause: the input DataFrame columns do not match the columns the model was trained on.\n\n"
                "If your saved model is a plain estimator (not a pipeline) trained on column names, set FEATURE_DEFS keys to match those names. "
                "If your training used a pipeline that includes transformers, consider saving the full pipeline (preprocessing + estimator) so predictions are consistent."
            )
            return

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Playoff Probability", value=f"{playoff_prob:.1%}")
        with col_res2:
            if playoff_prob > 0.80:
                st.success("🏆 Lock it in! This profile is a near-guaranteed playoff team.")
            elif playoff_prob > 0.50:
                st.warning("⚠️ Bubble Team. You're in the Play-In mix.")
            else:
                st.error("🎰 Lottery Bound. This statistical profile rarely makes the postseason.")

    # Optional: show the input dataframe for debugging
    with st.expander("Show input DataFrame (for debugging)"):
        st.write(input_df)

# ---------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------
def render_insights(df):
    st.header("📊 Behind the Model")
    tab1, tab2 = st.tabs(["Feature Importance", "Confusion Matrix"])

    with tab1:
        st.write("### Which Rankings Matter Most?")
        imp_path = os.path.join(VIS_DIR, "feature_importance_comparison.png")
        if os.path.exists(imp_path):
            st.image(imp_path, caption="Feature Importance (Ranks)", use_container_width=True)
        else:
            st.info("Feature importance plot not found in the visualizations folder.")

    with tab2:
        st.write("### Model Accuracy")
        cm_path = os.path.join(VIS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix (Test Set)", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found in the visualizations folder.")

    if df is not None:
        with st.expander("Show example rows from your dataset"):
            st.dataframe(df.head())

# ---------------------------------------------------------
# ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
