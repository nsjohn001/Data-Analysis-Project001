import streamlit as st
import pandas as pd
import joblib
import random

# Load model and supporting files
model = joblib.load("final_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
feature_names = joblib.load("feature_names.joblib")

st.title("🧠 Job Title Prediction App")
st.write("Enter your customer data to predict the most likely job title.")

# Create default or randomizable inputs
if st.button("🔀 Randomize Input"):
    city = random.choice(label_encoders["City"].classes_.tolist())
    country = random.choice(label_encoders["Country"].classes_.tolist())
    gender = random.choice(label_encoders["Gender"].classes_.tolist())
    email_domain = random.choice(label_encoders["Email Domain"].classes_.tolist())
    age_group = random.choice(label_encoders["Age Group"].classes_.tolist())
    telephone = ''.join(random.choices("0123456789", k=random.randint(10, 15)))
    age = random.randint(18, 60)
else:
    city = st.selectbox("City", label_encoders["City"].classes_.tolist())
    country = st.selectbox("Country", label_encoders["Country"].classes_.tolist())
    gender = st.selectbox("Gender", label_encoders["Gender"].classes_.tolist())
    email_domain = st.selectbox("Email Domain", label_encoders["Email Domain"].classes_.tolist())
    age_group = st.selectbox("Age Group", label_encoders["Age Group"].classes_.tolist())
    telephone = st.text_input("Telephone", "922970226547563")
    age = st.number_input("Age", min_value=10, max_value=100, value=25)

# Submit button
if st.button("Predict Job Title"):
    telephone_length = len(telephone)
    age_scaled = round(age / 58, 5)

    input_data = pd.DataFrame([{
        'City': city,
        'Country': country,
        'Gender': gender,
        'Email Domain': email_domain,
        'Age Group': age_group,
        'Telephone Length': telephone_length,
        'Age': age,
        'Age_Scaled': age_scaled
    }])

    # Encode categorical fields
    for col in label_encoders:
        if col in input_data.columns:
            input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data[feature_names]  # Match training feature order

    # 🧪 Debug: Show encoded input
    st.write("🧪 Encoded input used for prediction:")
    st.dataframe(input_data)

    # Predict job title
    prediction = model.predict(input_data)

    # Decode prediction
    job_title = label_encoders['Job Title'].inverse_transform(prediction)[0]
    st.success(f"🎯 Predicted Job Title: **{job_title}**")
