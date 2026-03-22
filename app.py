import streamlit as st
import numpy as np
import pickle

with open('cervical_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('cervical_cancer_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Cervical Cancer Risk Prediction")
st.write("Enter patient clinical details below to predict cervical cancer risk.")

st.header("Patient Information")
age = st.slider("Age", 10, 90, 30)
num_partners = st.number_input("Number of sexual partners", 0, 50, 1)
first_intercourse = st.number_input("Age at first sexual intercourse", 10, 50, 18)
num_pregnancies = st.number_input("Number of pregnancies", 0, 20, 1)

st.header("Lifestyle Factors")
smokes = st.selectbox("Smokes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
smokes_years = st.number_input("Smoking duration (years)", 0, 50, 0)
smokes_packs = st.number_input("Smoking packs per year", 0, 50, 0)

st.header("Medical History")
hormonal_contraceptives = st.selectbox("Hormonal Contraceptives", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
hormonal_years = st.number_input("Hormonal Contraceptives duration (years)", 0, 30, 0)
iud = st.selectbox("IUD", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
iud_years = st.number_input("IUD duration (years)", 0, 20, 0)
stds = st.selectbox("STDs", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
stds_number = st.number_input("Number of STD diagnoses", 0, 10, 0)

st.header("STD History")
condylomatosis = st.selectbox("Condylomatosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
cervical_condylomatosis = st.selectbox("Cervical Condylomatosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
vaginal_condylomatosis = st.selectbox("Vaginal Condylomatosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
vulvo_condylomatosis = st.selectbox("Vulvo-perineal Condylomatosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
syphilis = st.selectbox("Syphilis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pelvic_inflammatory = st.selectbox("Pelvic Inflammatory Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
genital_herpes = st.selectbox("Genital Herpes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
molluscum = st.selectbox("Molluscum Contagiosum", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
aids = st.selectbox("AIDS", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
hiv = st.selectbox("HIV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
hepatitis_b = st.selectbox("Hepatitis B", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
hpv = st.selectbox("HPV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
stds_diagnosis = st.number_input("Number of STD diagnoses (total)", 0, 10, 0)

st.header("Diagnostic Results")
dx_cancer = st.selectbox("Previous Cancer Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
dx_cin = st.selectbox("CIN Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
dx_hpv = st.selectbox("HPV Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
dx = st.selectbox("General Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
hinselmann = st.selectbox("Hinselmann test result", [0, 1], format_func=lambda x: "Positive" if x == 1 else "Negative")
schiller = st.selectbox("Schiller test result", [0, 1], format_func=lambda x: "Positive" if x == 1 else "Negative")
citology = st.selectbox("Citology result", [0, 1], format_func=lambda x: "Positive" if x == 1 else "Negative")

if st.button("Predict Cancer Risk"):
    input_data = np.array([[age, num_partners, first_intercourse, num_pregnancies,
                            smokes, smokes_years, smokes_packs, hormonal_contraceptives,
                            hormonal_years, iud, iud_years, stds, stds_number,
                            condylomatosis, cervical_condylomatosis, vaginal_condylomatosis,
                            vulvo_condylomatosis, syphilis, pelvic_inflammatory,
                            genital_herpes, molluscum, aids, hiv, hepatitis_b, hpv,
                            stds_diagnosis, dx_cancer, dx_cin, dx_hpv, dx,
                            hinselmann, schiller, citology]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High Risk — Cancer Detected (Confidence: {probability*100:.1f}%)")
        st.write("This patient is recommended for immediate clinical follow-up and may be a candidate for radiotherapy.")
    else:
        st.success(f"Low Risk — No Cancer Detected (Confidence: {(1-probability)*100:.1f}%)")
        st.write("This patient shows low risk. Regular screening is recommended.")