# app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# -------------------------------
# Load Model and Data
# -------------------------------
log_model = pickle.load(open("logistic_model.pkl", "rb"))
le_dict = pickle.load(open("label_encoders.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

df = pd.read_csv("HR Dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("💼 Dashboard")
page = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"])

# -------------------------------
# DASHBOARD PAGE
# -------------------------------
# -------------------------------
# DASHBOARD PAGE
# -------------------------------
if page == "Dashboard":
    st.title("Employee Attrition Dashboard")

    # KPIs
    attrited_df = df[df["Attrition"] == "Yes"]
    attrition_rate = len(attrited_df) / len(df) * 100
    attrition_count = len(attrited_df)
    avg_income = df["MonthlyIncome"].mean()

    # New KPI: Avg Years at Company (Attrited vs Non-Attrited)
    avg_years_attrited = df[df["Attrition"] == "Yes"]["YearsAtCompany"].mean()
    avg_years_stayed = df[df["Attrition"] == "No"]["YearsAtCompany"].mean()

    # Display KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("📉 Attrition Rate", f"{attrition_rate:.1f}%")
    kpi2.metric("🔁 Attrition Count", attrition_count)
    kpi3.metric("⏳ Avg Years (Attrited)", f"{avg_years_attrited:.1f} yrs")
    kpi4.metric("💰 Avg Monthly Income (Overall)", f"${avg_income:,.0f}")

    # Extra KPI below if you want income too
    st.markdown("---")


    theme = "plotly_white"

    # Chart layout
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(df, names="Attrition", title="Employee Attrition Rate",
                         hole=0.4, template=theme, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_gender = px.bar(df, x="Gender", color="Attrition", barmode="group",
                            title="Attrition by Gender", template=theme,
                            color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_gender, use_container_width=True)

        # ✅ New: Years at Company Attrition
        fig_years = px.histogram(df, x="YearsAtCompany", color="Attrition", barmode="group",
                                 title="Employee Attrition by Years at Company", template=theme,
                                 color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig_years, use_container_width=True)

    with col2:
        fig_marital = px.pie(df, names="MaritalStatus", color="Attrition",
                             title="Attrition by Marital Status", hole=0.4,
                             template=theme, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_marital, use_container_width=True)

        fig_travel = px.histogram(df, x="BusinessTravel", color="Attrition", barmode="group",
                                  title="Attrition by Business Travel", template=theme,
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_travel, use_container_width=True)

        fig_income = px.box(df, x="JobRole", y="MonthlyIncome", color="Attrition",
                            title="Monthly Income by Job Role", template=theme,
                            color_discrete_sequence=px.colors.qualitative.D3)
        st.plotly_chart(fig_income, use_container_width=True)

# -------------------------------
# PREDICTION PAGE
# -------------------------------
else:
    st.subheader("🔮 Predict Employee Attrition (Logistic Regression)")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 60, 30)
            gender = st.selectbox("Gender", df["Gender"].unique())
            job_role = st.selectbox("Job Role", df["JobRole"].unique())
            dept = st.selectbox("Department", df["Department"].unique())
            business_travel = st.selectbox("Business Travel", df["BusinessTravel"].unique())
            education_field = st.selectbox("Education Field", df["EducationField"].unique())

        with col2:
            monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
            overtime = st.selectbox("OverTime", df["OverTime"].unique())
            years_at_company = st.number_input("Years at Company", 0, 40, 5)
            total_working_years = st.number_input("Total Working Years", 0, 40, 10)
            environment_satisfaction = st.selectbox(
                "Environment Satisfaction",
                sorted(df["EnvironmentSatisfaction"].dropna().unique())
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "JobRole": [job_role],
            "Department": [dept],
            "BusinessTravel": [business_travel],
            "EducationField": [education_field],
            "MonthlyIncome": [monthly_income],
            "OverTime": [overtime],
            "YearsAtCompany": [years_at_company],
            "TotalWorkingYears": [total_working_years],
            "EnvironmentSatisfaction": [environment_satisfaction]
        })

        # Encode categorical variables
        for col in input_data.columns:
            if col in le_dict:
                input_data[col] = le_dict[col].transform(input_data[col])

        # Add any missing features
        for col in feature_columns:
            if col not in input_data:
                input_data[col] = 0

        input_data = input_data[feature_columns]

        # Make prediction
        pred = log_model.predict(input_data)[0]
        proba = log_model.predict_proba(input_data)[0][1]

        st.write("---")
        if pred == 1:
            st.error(f"⚠ Employee is likely to *Leave* (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Employee is likely to *Stay* (Probability: {1 - proba:.2f})")

           

