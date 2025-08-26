import streamlit as st
import pandas as pd
import xgboost==1.7.6 as xgb

# -------------------------
# Load model
# -------------------------
model = xgb.XGBClassifier()
model.load_model("heat_stress_multimodal_xgb.json")

# Feature groups
weather_features = ["air_temp", "humidity", "wind_speed", "solar_rad", "THI"]
sensor_features = ["rumen_temp", "resp_rate", "heart_rate", "rumination", "milk_yield", "cortisol", "NEFA"]
image_features = ["skin_temp_avg", "skin_temp_max", "panting_score_cnn"]
features = weather_features + sensor_features + image_features

st.title("🐄 Cattle Heat Stress Prediction")
st.write("Upload cow sensor + weather + image data to predict heat stress events.")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)

    missing = [col for col in features if col not in df_new.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
    else:
        # Run predictions
        df_new["heat_stress_prob"] = model.predict_proba(df_new[features])[:, 1]
        df_new["heat_stress_pred"] = model.predict(df_new[features])

        st.success("✅ Predictions complete!")
        st.dataframe(df_new.head())

        # Download button
        csv = df_new.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")
