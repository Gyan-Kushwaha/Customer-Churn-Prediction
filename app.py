import streamlit as st
import pandas as pd
import joblib
import shap

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

model_artifact = joblib.load("customer_churn_model.pkl")

pipeline = model_artifact["pipeline"]
threshold = model_artifact["threshold"]
background = model_artifact["background"]

explainer = shap.LinearExplainer(
    pipeline.named_steps["model"],
    background
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer information to predict the likelihood of churn.")

st.sidebar.title("Model Information")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write(f"Decision threshold: {threshold}")
st.sidebar.write("ROC-AUC: 0.836")
st.sidebar.write("F1 Score: 0.611")

st.divider()

st.subheader("👤 Customer Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])

with col3:
    partner = st.selectbox("Partner", ["Yes", "No"])

with col4:
    dependents = st.selectbox("Dependents", ["Yes", "No"])


st.subheader("📱 Services")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


st.subheader("💳 Account Information")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0
    )


st.divider()

if st.button("🔮 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    probability = pipeline.predict_proba(input_data)[0, 1]

    transformed_input = pipeline.named_steps["preprocessor"].transform(input_data)

    shap_values = explainer.shap_values(transformed_input)[0]

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    feature_importance = {}

    for column in input_data.columns:

        if f"num__{column}" in feature_names:
            index = list(feature_names).index(f"num__{column}")
            feature_importance[column] = shap_values[index]

        else:
            prefix = f"cat__{column}_"

            indices = [
                i for i, name in enumerate(feature_names)
                if name.startswith(prefix)
            ]

            feature_importance[column] = shap_values[indices].sum()

    feature_importance = pd.DataFrame(
        list(feature_importance.items()),
        columns=["Feature", "SHAP"]
    )

    feature_importance["Absolute_SHAP"] = feature_importance["SHAP"].abs()

    top_features = feature_importance.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).head(8)


    prediction = "Yes" if probability >= threshold else "No"

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        if prediction == "Yes":
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is unlikely to churn")

    st.progress(float(probability))

    st.caption(
        f"The prediction uses a decision threshold of {threshold:.2f}."
    )

    st.subheader("📊 Feature Contributions")

    chart_data = top_features.copy()

    chart_data["Feature"] = chart_data["Feature"].apply(
        lambda x: f"{x} = {input_data.iloc[0][x]}"
    )

    chart_data = chart_data.sort_values("SHAP")

    chart_data = chart_data.set_index("Feature")[["SHAP"]]

    st.bar_chart(chart_data)

    st.subheader("🔍 Why did the model make this prediction?")

    for _, row in top_features.iterrows():

        feature = row["Feature"]
        shap_value = row["SHAP"]
        value = input_data.iloc[0][feature]

        if shap_value > 0:
            st.write(
                f"🔴 **{feature} = {value}** → contributes toward churn"
            )
        else:
            st.write(
                f"🟢 **{feature} = {value}** → contributes toward no churn"
            )