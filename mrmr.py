import streamlit as st
import pandas as pd
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

st.title("mRMR Feature Selection using MATLAB fsrmrmr")

# ------------------ Upload Section ------------------
uploaded_file = st.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ------------------ Target Column ------------------
    target_col = st.selectbox("Select Target Column (Y)", df.columns)

    if target_col:
        # ------------------ Input Feature Selection ------------------
        possible_features = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect("Select Predictor Features (X)", possible_features, default=possible_features)

        if selected_features:
            st.write(f"Selected Target: {target_col}")
            st.write(f"Selected Features: {', '.join(selected_features)}")

            if st.button("Run fsrmrmr (MATLAB)"):
                st.info("Starting MATLAB engine...")
                eng = matlab.engine.start_matlab()

                # Extract X and y
                X = df[selected_features]
                y = df[target_col]

                # Convert DataFrame to MATLAB double
                X_mat = matlab.double(X.values.tolist())
                y_mat = matlab.double(y.values.tolist())

                # ------------------ Run fsrmrmr ------------------
                try:
                    idx, scores = eng.fsrmrmr(X_mat, y_mat, nargout=2)

                    # Convert MATLAB doubles to Python
                    idx = [int(i) for i in np.array(idx).flatten()]
                    scores = np.array(scores).flatten().tolist()

                    feature_names = X.columns
                    selected_features_ranked = [feature_names[i-1] for i in idx]  # MATLAB indices are 1-based

                    results = pd.DataFrame({
                        "Feature": selected_features_ranked,
                        "Index": idx,
                        "Score": scores
                    })

                    st.success("fsrmrmr completed successfully ✅")
                    st.write("### mRMR Feature Ranking with Scores")
                    st.dataframe(results)

                    # ------------------ Plot ------------------
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(results["Feature"], results["Score"], color="skyblue")
                    ax.set_xlabel("mRMR Score")
                    ax.set_ylabel("Feature")
                    ax.set_title("mRMR Feature Importance (fsrmrmr)")
                    ax.invert_yaxis()  # highest score on top
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error running fsrmrmr: {e}")
