import streamlit as st
import pickle
import numpy as np

# ── Load model and encoder ──────────────────────────────────────────────────
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('crop_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")

st.title("🌾 Crop Recommendation System")
st.markdown("Enter your soil and weather conditions to get the best crop recommendation.")
st.markdown("---")

# ── Input fields ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🪨 Soil Nutrients")
    N         = st.slider("Nitrogen (N)", 0, 140, 90)
    P         = st.slider("Phosphorus (P)", 5, 145, 42)
    K         = st.slider("Potassium (K)", 5, 205, 43)
    ph        = st.slider("Soil pH", 3.5, 10.0, 6.5)

with col2:
    st.subheader("🌤️ Weather Conditions")
    temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
    humidity    = st.slider("Humidity (%)", 14.0, 100.0, 71.0)
    rainfall    = st.slider("Rainfall (mm)", 20.0, 300.0, 103.0)

# ── Predict button ──────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🌱 Recommend Crop", type="primary"):

    # ✅ Correct order: N, P, K, temperature, humidity, ph, rainfall
    # (same order as training dataset columns)
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction  = model.predict(input_data)[0]
    crop_name   = le.inverse_transform([prediction])[0]
    proba       = model.predict_proba(input_data)[0]
    confidence  = round(max(proba) * 100, 2)

    # Show result
    st.success(f"✅ Recommended Crop:  **{crop_name.upper()}**")
    st.metric("Model Confidence", f"{confidence}%")

    # Top 3 crops
    st.markdown("---")
    st.subheader("🏆 Top 3 Crop Suggestions")
    top3_idx   = np.argsort(proba)[::-1][:3]
    top3_crops = le.inverse_transform(top3_idx)
    top3_probs = proba[top3_idx]

    for i, (crop, prob) in enumerate(zip(top3_crops, top3_probs), 1):
        st.progress(int(prob * 100), text=f"{i}. {crop.capitalize()} — {prob*100:.1f}%")

    # Soil tips
    st.markdown("---")
    st.subheader("📋 Soil Health Tips")
    if N < 40:    st.warning("• Nitrogen is low — consider adding urea or compost")
    if P < 20:    st.warning("• Phosphorus is low — apply DAP fertilizer")
    if K < 20:    st.warning("• Potassium is low — apply MOP fertilizer")
    if ph < 5.5:  st.warning("• Soil is too acidic (pH < 5.5) — apply lime")
    if ph > 7.5:  st.warning("• Soil is too alkaline (pH > 7.5) — add sulfur")
    if rainfall < 50: st.info("• Low rainfall — ensure proper irrigation")
    if humidity > 90: st.info("• Very high humidity — watch for fungal diseases")

    if N >= 40 and P >= 20 and K >= 20 and 5.5 <= ph <= 7.5:
        st.success("• Soil nutrients and pH are in good range! 🎉")