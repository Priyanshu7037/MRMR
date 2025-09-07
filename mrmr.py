import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

st.set_page_config(layout="wide", page_title="mRMR Feature Selection")
st.title("MRMR Feature Selection ")

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

    # Target column
    target_col = st.selectbox("Select Target Column (Y)", df.columns)

    if target_col:
        possible_features = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect("Select Predictor Features (X)", possible_features, default=possible_features)

        if selected_features:
            N_features = st.number_input(
                "Number of top features to select (N)", 
                min_value=1, 
                max_value=len(selected_features), 
                value=min(10, len(selected_features)), 
                step=1
            )

            if st.button("Run mRMR", type="primary"):
                st.info("Computing mRMR scores...")

                # ------------------ Discretize ------------------
                def discretize_data(data_series, n_bins=10):
                    if pd.api.types.is_numeric_dtype(data_series):
                        try:
                            return pd.qcut(data_series, n_bins, duplicates='drop').cat.codes
                        except:
                            return pd.cut(data_series, n_bins, duplicates='drop', include_lowest=True).cat.codes
                    else:
                        return data_series.astype('category').cat.codes

                X = df[selected_features].copy()
                y = df[target_col].copy()
                X_disc = X.apply(discretize_data, n_bins=10)
                y_disc = discretize_data(y, n_bins=10)

                # ------------------ mRMR ------------------
                def mrmr_select(X_disc, y_disc, n_features):
                    features = list(X_disc.columns)
                    n_total = len(features)

                    relevance = {col: mutual_info_score(X_disc[col], y_disc) for col in features}

                    mi_matrix = np.zeros((n_total, n_total))
                    for i in range(n_total):
                        for j in range(i+1, n_total):
                            val = mutual_info_score(X_disc[features[i]], X_disc[features[j]])
                            mi_matrix[i, j] = val
                            mi_matrix[j, i] = val

                    selected_idx = []
                    scores, relevances, redundancies = [], [], []
                    remaining = list(range(n_total))

                    for _ in range(n_features):
                        if not selected_idx:
                            best_idx = remaining[np.argmax([relevance[features[i]] for i in remaining])]
                            score = relevance[features[best_idx]]
                            redund = 0.0
                        else:
                            vals = []
                            rels = []
                            reds = []
                            for idx in remaining:
                                redund = np.mean([mi_matrix[idx, s] for s in selected_idx])
                                rel = relevance[features[idx]]
                                vals.append(rel - redund)
                                rels.append(rel)
                                reds.append(redund)
                            pos = np.argmax(vals)
                            best_idx = remaining[pos]
                            score = vals[pos]
                            rel = rels[pos]
                            redund = reds[pos]

                        selected_idx.append(best_idx)
                        remaining.remove(best_idx)

                        scores.append(score)
                        relevances.append(relevance[features[best_idx]])
                        redundancies.append(redund)

                    return [features[i] for i in selected_idx], scores, relevances, redundancies

                # Run selection
                feats, scores, rels, reds = mrmr_select(X_disc, y_disc, N_features)

                # ------------------ Results ------------------
                results = pd.DataFrame({
                    "Feature": feats,
                    "Rank": range(1, len(feats)+1),
                    "Relevance (MI with target)": rels,
                    "Redundancy (Avg MI with selected)": reds,
                    "mRMR_Score (Relevance - Redundancy)": scores
                })

                st.success("✅ mRMR completed")
                st.dataframe(results)

                # ------------------ Plot ------------------
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(results["Feature"], results["mRMR_Score (Relevance - Redundancy)"], color="skyblue")
                ax.set_xlabel("mRMR Score")
                ax.set_title("mRMR Feature Ranking")
                ax.invert_yaxis()
                st.pyplot(fig)

