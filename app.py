import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


model = pickle.load(open('model_uas.pkl', 'rb'))

sc = StandardScaler()


categories_sex = [['female', 'male']]
categories_smoker = [['no', 'yes']]


def preprocess_input(age, sex, bmi, children, smoker):
 
  oe_sex = OrdinalEncoder(categories=categories_sex)
  sex_encoded = oe_sex.fit_transform([[sex]])[0][0]
  
  oe_smoker = OrdinalEncoder(categories=categories_smoker)
  smoker_encoded = oe_smoker.fit_transform([[smoker]])[0][0]
  
  
  input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded]])
  
  
  
  
  features = np.array([[age, sex_encoded, bmi, children, smoker_encoded]])
  input_scaled = sc.fit_transform(features)
  

  return input_scaled 


st.title("Prediksi Biaya Asuransi Kesehatan")
st.write("NIM: 2019230152")
st.write("Nama: Dwi Pramana Putra")

# Input Form
age = st.slider("Umur", 18, 80)
sex = st.selectbox("Jenis Kelamin", ["female", "male"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0)
children = st.selectbox("Jumlah Anak", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Perokok", ["no", "yes"])

# Button prediksi
if st.button("Prediksi Biaya Asuransi"):
  input_processed = preprocess_input(age, sex, bmi, children, smoker)
  prediction = model.predict(input_processed)[0]
  st.success(f"Prediksi Biaya Asuransi: ${prediction:.2f}")