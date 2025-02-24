import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
from sentence_transformers import SentenceTransformer

# ------------------ Platform UI ------------------
st.set_page_config(page_title="ML Experiment Platform", layout="wide")
st.title("🔬 Machine Learning Experiment Platform")

# Sidebar for Data Upload
st.sidebar.header("📂 Upload Your Data")
file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

# Smart Text Box for Objective Input
st.sidebar.subheader("🎯 Define Your Objective")
objective_text = st.sidebar.text_area("Describe your objective for this data analysis")


def clean_column_names(df):
    """Cleans column names by converting to lowercase, removing symbols, and standardizing units."""
    df.columns = [re.sub(r'[^a-zA-Z0-9_%]+', '_', col.lower().strip()) for col in df.columns]
    return df


if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df = clean_column_names(df)
    st.sidebar.success("✅ Data Loaded Successfully!")
    st.write("### Raw Data Preview:")
    st.dataframe(df)

    # Display User Objective
    if objective_text:
        st.write("### User Objective:")
        st.write(objective_text)

    # Data Cleaning
    st.write("### 🔍 Data Cleaning")
    
    # Handle missing values, standardize data types
    imputer = SimpleImputer(strategy='mean')
    df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    st.write("Cleaned Data Preview:")
    st.dataframe(df_clean)

    # Select Target Variables (Multi-target support)
    target_cols = st.sidebar.multiselect("Select Target Variables", df_clean.columns)

    # Select Features (Exclude selected targets)
    feature_cols = st.sidebar.multiselect(
        "Select Features",
        [col for col in df_clean.columns if col not in target_cols]
    )

    if feature_cols and target_cols:
        X = df_clean[feature_cols]
        y = df_clean[target_cols]

        # Convert objective text into embeddings for smarter training
        if objective_text:
            model_embedder = SentenceTransformer("all-MiniLM-L6-v2")
            objective_embedding = model_embedder.encode([objective_text])
            objective_embedding = np.tile(objective_embedding, (X.shape[0], 1))
            X = np.hstack((X, objective_embedding))

        # Save input shape to maintain consistency
        st.session_state["input_shape"] = X.shape[1]

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize Data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.sidebar.subheader("⚙️ Model Selection & Training")

        # Train MultiOutputRegressor with Gradient Boosting
        if st.sidebar.button("Train Model"):
            model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5))
            model.fit(X_train_scaled, y_train)
            st.session_state['model'] = model  # Store model in session state
            st.sidebar.success("✅ Model Trained Successfully!")

        # Experiment Generation
        st.sidebar.subheader("🧪 Generate Realistic Experiments")
        num_experiments = st.sidebar.number_input("Number of Experiments", min_value=1, max_value=10000, value=10)

        if st.sidebar.button("Generate Experiments"):
            if 'model' in st.session_state:
                model = st.session_state['model']

                # Ensure correct dimensions for generated data
                X_train_np = np.array(X_train)  # Convert to NumPy
                X_min, X_max = X_train_np.min(axis=0), X_train_np.max(axis=0)

                synthetic_data = resample(X_train_np, n_samples=num_experiments, random_state=42)
                noise = np.random.normal(0, 0.02, synthetic_data.shape)
                synthetic_data = synthetic_data + noise

                # Ensure values are within realistic constraints
                synthetic_data = np.clip(synthetic_data, X_min, X_max)

                # Predict Conductivity & Resistance
                synthetic_targets = model.predict(scaler.transform(synthetic_data))

                # Create DataFrame for Generated Experiments
                generated_experiments = pd.DataFrame(synthetic_data[:, :len(feature_cols)], columns=feature_cols)

                # Ensure targets stay within valid ranges
                for i, target in enumerate(target_cols):
                    generated_experiments[target] = np.clip(
                        synthetic_targets[:, i],
                        y_train[target].min(),
                        y_train[target].max()
                    )

                st.write("### ✅ All Newly Generated Experiments (Sorted)")
                st.dataframe(generated_experiments.sort_values(by=target_cols, ascending=True))

                # **NEW: Select Best Experiment Based on User Preference**
                st.subheader("⭐ Choose the Best Experiment Criteria")
                best_criteria = st.selectbox(
                    "Select Optimization Preference",
                    ["Lowest Value", "Highest Value", "Balanced Optimization"]
                )

                selected_variable = st.selectbox("Select Variable to Optimize", target_cols)

                if best_criteria == "Lowest Value":
                    best_experiment = generated_experiments.loc[generated_experiments[selected_variable].idxmin()]
                elif best_criteria == "Highest Value":
                    best_experiment = generated_experiments.loc[generated_experiments[selected_variable].idxmax()]
                else:
                    # Balanced approach: Normalizing and picking the best compromise
                    normalized_data = (generated_experiments[target_cols] - generated_experiments[target_cols].min()) / \
                                      (generated_experiments[target_cols].max() - generated_experiments[target_cols].min())
                    best_experiment = generated_experiments.loc[normalized_data.sum(axis=1).idxmin()]

                st.write("### 🏆 Best Selected Experiment Based on Your Choice")
                st.dataframe(best_experiment.to_frame().T)

                # Download Option
                st.download_button(
                    label="📥 Download Generated Experiments",
                    data=generated_experiments.to_csv(index=False),
                    file_name="generated_experiments.csv",
                    mime="text/csv"
                )
            else:
                st.error("⚠ Please train the model before generating experiments.")

        # Download Model Results
        st.sidebar.subheader("📥 Download Model Results")
        if 'model' in st.session_state and 'input_shape' in st.session_state:
            expected_features = st.session_state["input_shape"]
            if X_test_scaled.shape[1] != expected_features:
                st.error(f"⚠ Feature mismatch! Expected {expected_features} features but got {X_test_scaled.shape[1]}.")
            else:
                y_pred = st.session_state['model'].predict(X_test_scaled)
                results_df = pd.DataFrame(X_test, columns=feature_cols)
                for i, target in enumerate(target_cols):
                    results_df[f"Predicted_{target}"] = y_pred[:, i]
                    results_df[f"Actual_{target}"] = y_test.reset_index(drop=True)[target]

                st.write("### 📊 Model Results")
                st.dataframe(results_df)

                st.download_button(
                    label="📥 Download Model Results",
                    data=results_df.to_csv(index=False),
                    file_name="model_results.csv",
                    mime="text/csv"
                )

st.sidebar.info("🚀 Upload a dataset and start experimenting!")
