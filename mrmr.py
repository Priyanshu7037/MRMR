import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# ------------------ Streamlit Page Config ------------------ #
st.set_page_config(layout="wide", page_title="MRMR Feature Selection (Regression)")
st.title("MRMR Feature Selection (Regression)")

st.markdown("""
This app performs **Minimum Redundancy Maximum Relevance (mRMR)** feature selection for regression.  

- **Relevance**: Mutual Information (MI) between feature and target.  
- **Redundancy**: Average absolute correlation with already-selected features.  
- **Selection**: Pick features that maximize **Relevance − Redundancy**.  
""")

# ------------------ Dataset Choice ------------------ #
data_choice = st.radio(
    "Choose dataset source",
    ["Upload File", "Use Demo Dataset"],
    horizontal=True
)

df = None

if data_choice == "Upload File":
    uploaded_file = st.file_uploader("Upload your dataset (CSV/XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
else:
    # ----------- Generate Synthetic Demo Dataset ----------- #
    np.random.seed(42)
    n_samples = 300
    n_features = 15

    # Randomly choose 3–5 "true" features
    n_true = np.random.randint(3, 6)
    true_indices = np.random.choice(n_features, n_true, replace=False)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Random coefficients for true features
    coefs = np.random.uniform(-3, 3, size=n_true)

    # Target = linear combo of chosen features + noise
    y = X[:, true_indices] @ coefs + 0.5 * np.random.randn(n_samples)

    # Combine into dataframe
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_features)])
    df["Target"] = y

    st.info(f"✅ Using demo dataset (Target depends on {n_true} true features: {[f'X{i+1}' for i in true_indices]})")

# ------------------ Run if data exists ------------------ #
if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select target
    target = st.selectbox("Select target column (Y)", df.columns, index=len(df.columns)-1)
    features = [col for col in df.columns if col != target]

    X = df[features].fillna(0).values
    y = df[target].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Relevance (MI)
    relevance = mutual_info_regression(X_scaled, y, random_state=0)

    # Redundancy (Correlation)
    corr_matrix = np.corrcoef(X_scaled, rowvar=False)
    redundancy = np.abs(corr_matrix)

    # mRMR selection
    selected = []
    scores, relevances, redundancies = {}, {}, {}
    k = st.slider("Number of features to select", 1, len(features), min(5, len(features)))

    for _ in range(k):
        best_score = -np.inf
        best_feature = None
        best_rel, best_red = None, None

        for i, feat in enumerate(features):
            if feat in selected:
                continue

            rel = relevance[i]
            if selected:
                red = np.mean([redundancy[i, features.index(s)] for s in selected])
            else:
                red = 0

            score = rel - red
            if score > best_score:
                best_score = score
                best_feature = feat
                best_rel = rel
                best_red = red

        selected.append(best_feature)
        scores[best_feature] = best_score
        relevances[best_feature] = best_rel
        redundancies[best_feature] = best_red

    # Results Table
    st.write("### Selected Features (mRMR)")
    result_df = pd.DataFrame({
        "Feature": selected,
        "Rank": range(1, len(selected)+1),
        "Relevance (MI with Target)": [relevances[f] for f in selected],
        "Redundancy (Avg Corr with selected)": [redundancies[f] for f in selected],
        "mRMR Score (Rel - Red)": [scores[f] for f in selected]
    })
    st.dataframe(result_df)

    # Plot Scores
    fig, ax = plt.subplots()
    ax.bar(result_df["Feature"], result_df["mRMR Score (Rel - Red)"], color="skyblue")
    ax.set_title("mRMR Scores for Selected Features")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Explanation
    
