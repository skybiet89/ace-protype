import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Streamlit App Title
# -----------------------------
st.set_page_config(page_title="ACE Prototype", layout="wide")
st.title("🧠 ACE Buyer Prediction Prototype")

# -----------------------------
# Generate / Upload Data
# -----------------------------
st.sidebar.header("Data Options")
data_source = st.sidebar.radio("Choose data source:", ["Generate Sample", "Upload CSV"])

if data_source == "Generate Sample":
    np.random.seed(42)
    buyers_df = pd.DataFrame({
        "age": np.random.randint(18, 60, 100),
        "income": np.random.randint(20000, 120000, 100),
        "previous_purchases": np.random.randint(0, 10, 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "target": np.random.choice([0, 1], 100)   # binary classification
    })

else:  # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        buyers_df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# Save in session state
st.session_state['buyers_df'] = buyers_df

# -----------------------------
# Feature Engineering
# -----------------------------
# Encode categorical features
buyers_df = pd.get_dummies(buyers_df, drop_first=True)

# Separate features and target
if "target" in buyers_df.columns:
    X = buyers_df.drop("target", axis=1)
    y = buyers_df["target"]
else:
    st.error("❌ No 'target' column found in data. Please include a target column.")
    st.stop()

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# Store model + features in session
# -----------------------------
st.session_state['model'] = model
st.session_state['feat_cols'] = list(X.columns)

# -----------------------------
# Results
# -----------------------------
st.subheader("🔍 Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Prediction Demo
# -----------------------------
st.subheader("🎯 Try a Prediction")

# Dynamically create inputs
input_data = {}
for col in st.session_state['feat_cols']:
    if buyers_df[col].dtype in [np.int64, np.float64]:
        input_data[col] = st.number_input(f"{col}", float(buyers_df[col].min()), float(buyers_df[col].max()), float(buyers_df[col].mean()))
    else:
        input_data[col] = st.selectbox(f"{col}", buyers_df[col].unique())

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Fix missing columns
for col in st.session_state['feat_cols']:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[st.session_state['feat_cols']]

# Prediction
if st.button("Predict"):
    proba = st.session_state['model'].predict_proba(input_df)[0]
    st.write("Prediction Probability:", proba)
    st.success(f"Predicted Buyer Class: {np.argmax(proba)}")
