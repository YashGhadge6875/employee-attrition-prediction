# -------------------------------
# Imports
# -------------------------------
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# -------------------------------
# Load Models, Encoders, and Data
# -------------------------------
log_model = pickle.load(open("logistic_model.pkl", "rb"))
le_dict = pickle.load(open("label_encoders.pkl", "rb"))
feature_cols = pickle.load(open("feature_columns.pkl", "rb"))

df = pd.read_csv("HR Dataset.csv")  # Original dataset for dashboard

# Encode categorical columns for dashboard visualizations
for col, le in le_dict.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

# -------------------------------
# Page Selection
# -------------------------------
page = st.sidebar.selectbox("Choose Page", ["Dashboard", "Prediction"])

# -------------------------------
# Page Config & Custom CSS
# -------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight:600;
    color:#2F4F4F;
}
.metric-card {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# DASHBOARD PAGE
# -------------------------------
if page == "Dashboard":
    st.image("HR image.png", use_container_width=True)
    st.title("📊 Employee Attrition Dashboard")
    st.markdown("<div class='big-font'>A bright and interactive dashboard to analyze attrition trends</div>", unsafe_allow_html=True)

    # KPIs
    attrited_df = df[df["Attrition"] == 1]  # 1 = Yes after encoding
    attrition_rate = len(attrited_df) / len(df) * 100
    attrition_count = len(attrited_df)
    avg_income = df["MonthlyIncome"].mean()

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"<div class='metric-card'>📉<br>Attrition Rate<br><h3>{attrition_rate:.1f}%</h3></div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"<div class='metric-card'>🔁<br>Attrition Count<br><h3>{attrition_count}</h3></div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='metric-card'>💰<br>Avg Monthly Income<br><h3>${avg_income:,.0f}</h3></div>", unsafe_allow_html=True)

    st.markdown("---")
    theme = "plotly_white"

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(df, names="Attrition", title="Employee Attrition Rate",
                         hole=0.4, template=theme, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_gender = px.bar(df, x="Gender", color="Attrition", barmode="group",
                            title="Attrition by Gender", template=theme,
                            color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_gender, use_container_width=True)

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
    st.info("Fill out employee details below to predict attrition likelihood.")

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

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        # Prepare input dataframe
        input_dict = {
            "Age": age,
            "Gender": le_dict["Gender"].transform([gender])[0],
            "JobRole": le_dict["JobRole"].transform([job_role])[0],
            "Department": le_dict["Department"].transform([dept])[0],
            "BusinessTravel": le_dict["BusinessTravel"].transform([business_travel])[0],
            "EducationField": le_dict["EducationField"].transform([education_field])[0],
            "MonthlyIncome": monthly_income,
            "OverTime": le_dict["OverTime"].transform([overtime])[0],
            "YearsAtCompany": years_at_company,
            "TotalWorkingYears": total_working_years,
            "EnvironmentSatisfaction": environment_satisfaction
        }

        input_data = pd.DataFrame([input_dict], columns=feature_cols)

        pred = log_model.predict(input_data)[0]
        proba = log_model.predict_proba(input_data)[0][1]

        st.write("---")
        if pred == 1:
            st.error(f"⚠️ Employee is likely to **Leave** (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Employee is likely to **Stay** (Probability: {1 - proba:.2f})")

