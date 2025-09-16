# app.py
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# Load Model and Dataset
# -------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Load model
model = pickle.load(open("logistic_model_new.pkl", "rb"))
df = pd.read_csv("HR Dataset.csv")

st.title("💼 Employee Attrition Prediction Dashboard")

# -------------------------------
# Tabs for Navigation
# -------------------------------
tab1, tab2 = st.tabs(["📊 EDA Dashboard", "🔮 Prediction"])

# -------------------------------
# TAB 1: EDA Dashboard
# -------------------------------
with tab1:
    st.subheader("Exploratory Data Analysis")

    # Sidebar filters
    st.sidebar.header("🔎 Filter Data")
    gender_filter = st.sidebar.multiselect("Select Gender", df["Gender"].unique(), default=df["Gender"].unique())
    dept_filter = st.sidebar.multiselect("Select Department", df["Department"].unique(), default=df["Department"].unique())

    df_filtered = df[(df["Gender"].isin(gender_filter)) & (df["Department"].isin(dept_filter))]

    # KPIs
    col1, col2, col3 = st.columns(3)
    attrition_rate = (df_filtered["Attrition"].value_counts(normalize=True).get("Yes", 0) * 100)
    avg_income = df_filtered["MonthlyIncome"].mean()
    avg_age = df_filtered["Age"].mean()
    avg_years = df_filtered["YearsAtCompany"].mean()

    col1.metric("Attrition %", f"{attrition_rate:.1f}%")
    col2.metric("Avg. Monthly Income", f"${avg_income:,.0f}")
    col3.metric("Avg. Tenure (Years)", f"{avg_years:.1f}")

    # -------------------------------
    # Charts
    # -------------------------------
    st.markdown("### 📊 Visual Insights")

    # Attrition Pie Chart
    fig_pie = px.pie(df_filtered, names="Attrition", title="Employee Attrition Rate", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Gender vs Attrition
    fig_gender = px.bar(df_filtered, x="Gender", color="Attrition", barmode="group",
                        title="Attrition by Gender")
    st.plotly_chart(fig_gender, use_container_width=True)

    # Business Travel vs Attrition
    fig_travel = px.histogram(df_filtered, x="BusinessTravel", color="Attrition", barmode="group",
                              title="Attrition by Business Travel")
    st.plotly_chart(fig_travel, use_container_width=True)

    # Education Field vs Attrition
    fig_edu = px.histogram(df_filtered, x="EducationField", color="Attrition", barmode="group",
                           title="Attrition by Education Field")
    st.plotly_chart(fig_edu, use_container_width=True)

    # Job Role vs Monthly Income
    fig_income = px.box(df_filtered, x="JobRole", y="MonthlyIncome", color="Attrition",
                        title="Monthly Income Distribution by Job Role")
    st.plotly_chart(fig_income, use_container_width=True)

    # Age vs Monthly Income Scatter
    fig_scatter = px.scatter(df_filtered, x="Age", y="MonthlyIncome", color="Attrition",
                             size="YearsAtCompany", hover_data=["JobRole"],
                             title="Age vs Income (Bubble size = Years at Company)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    df_encoded = pd.get_dummies(df_filtered, drop_first=True)
    corr = df_encoded.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                         title="Correlation Heatmap of HR Features")
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------
# TAB 2: Prediction
# -------------------------------
with tab2:
    st.subheader("🔮 Predict Employee Attrition")

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
            job_level = st.selectbox("Job Level", sorted(df["JobLevel"].unique()))
            environment_satisfaction = st.selectbox("Environment Satisfaction", sorted(df["EnvironmentSatisfaction"].unique()))

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

        # Encode with dummies
        input_encoded = pd.get_dummies(input_data)

        # Align with training columns
        train_cols = model.feature_names_in_
        for col in train_cols:
            if col not in input_encoded:
                input_encoded[col] = 0
        input_encoded = input_encoded[train_cols]

        # Prediction
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0][1]

        # Result Display
        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ Employee is likely to **leave**.\n\n🔹 Probability: {proba:.2f}")
        else:
            st.success(f"✅ Employee is likely to **stay**.\n\n🔹 Probability: {proba:.2f}")



