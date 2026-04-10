import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model, scaler = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Smart Study Assistant", page_icon="📚")

st.title("📚 Smart Study Assistant")
st.markdown("Predict and analyze your study performance 📊")
st.divider()

# Inputs
study_hours = st.slider("📖 Study Hours per Day", 0, 10, 5)
attendance = st.slider("🏫 Attendance (%)", 0, 100, 70)
previous_score = st.slider("📝 Previous Score (%)", 0, 100, 60)

# ---- GRAPH 1: Input Visualization ----
st.subheader("📊 Your Study Profile")

input_df = pd.DataFrame({
    "Feature": ["Study Hours", "Attendance", "Previous Score"],
    "Value": [study_hours, attendance, previous_score]
})

fig1, ax1 = plt.subplots()
ax1.bar(input_df["Feature"], input_df["Value"])
ax1.set_title("Your Inputs Overview")
st.pyplot(fig1)

# Prediction
if st.button("🔮 Predict Understanding"):
    input_data = np.array([[study_hours, attendance, previous_score]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    # Result
    if prediction[0] == 1:
        st.success("🎉 You will likely understand the topic!")
        st.balloons()
    else:
        st.error("⚠ You may struggle. Try revising more!")

    # ---- GRAPH 2: Confidence Chart ----
    st.subheader("📈 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Outcome": ["Struggle", "Understand"],
        "Probability": probability
    })

    fig2, ax2 = plt.subplots()
    ax2.bar(prob_df["Outcome"], prob_df["Probability"])
    ax2.set_title("Model Confidence")
    st.pyplot(fig2)

# ---- GRAPH 3: Study Recommendation ----
st.subheader("📉 Study Recommendation")

if study_hours < 4:
    st.warning("📖 Try increasing your study hours!")

if attendance < 60:
    st.warning("🏫 Improve your attendance!")

if previous_score < 50:
    st.warning("📝 Revise previous topics!")

st.divider()

# ---- Simple Analytics ----
st.subheader("📊 Quick Analytics")

avg_score = (study_hours * 10 + attendance + previous_score) / 3

st.metric("📈 Overall Performance Score", f"{avg_score:.2f}")

st.write("App is running ✅")