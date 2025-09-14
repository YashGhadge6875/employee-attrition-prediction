# app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ----------------- Load Dataset -----------------
df = pd.read_csv("HR Dataset.csv")

# ----------------- Sidebar -----------------
st.sidebar.title("⚡ Employee Attrition Dashboard")
menu = st.sidebar.radio("Navigate", ["EDA Dashboard", "Predict Attrition"])

# ----------------- EDA Dashboard -----------------
if menu == "EDA Dashboard":
    st.title("📊 Employee Attrition Analysis")

    # Pie Chart - Attrition Rate
    fig1 = px.pie(df, names="Attrition", title="Employee Attrition Rate", hole=0.4)
    st.plotly_chart(fig1)

    # Attrition by Gender
    fig2 = px.bar(df, x="Gender", color="Attrition", barmode="group",
                  title="Attrition by Gender")
    st.plotly_chart(fig2)

    # Attrition by Business Travel
    fig3 = px.bar(df, x="BusinessTravel", color="Attrition", barmode="group",
                  title="Attrition by Business Travel")
    st.plotly_chart(fig3)

    # Attrition by Education Field
    fig4 = px.bar(df, x="EducationField", color="Attrition", barmode="group",
                  title="Attrition by Education Field")
    st.plotly_chart(fig4)

# ----------------- Prediction -----------------
elif menu == "Predict Attrition":
    st.title("🤖 Predict Employee Attrition (Logistic Regression)")

    # Load Logistic Regression Model
    log_model = pickle.load(open("models/logistic_model.pkl", "rb"))

    st.write("### Enter Employee Details:")

    # Input form dynamically
    input_data = {}
    for col in df.drop("Attrition", axis=1).columns:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.number_input(
                col, float(df[col].min()), float(df[col].max()), float(df[col].mean())
            )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # One-hot encode & align
        final_df = pd.get_dummies(input_df).reindex(columns=df.drop("Attrition", axis=1).columns, fill_value=0)

        prediction = log_model.predict(final_df)[0]

        if prediction == 1:
            st.error("⚠️ This employee is likely to **Leave (Attrition = Yes)**")
        else:
            st.success("✅ This employee is likely to **Stay (Attrition = No)**")
