from io import BytesIO
import pathlib
import streamlit as st
from fastai.data.all import *
from fastai.vision.all import *
from PIL import Image
from fastai.learner import load_learner

from service import set_background

import json

# Open the JSON file and load its contents
with open('file.json') as json_file:
    data = json.load(json_file)

st.title("Wound Classification App")

st.write("Upload an image and a trained Ai model will predict its class with giving First Aid Advice") 

def label_func(fname):
    categories = ["Class_"+ str(i) for i in range(1,8)]
    category = fname.parts[-2]  # Extract the category name from the path
    return category if category in labels else "unknown"

import pathlib
pathlib.PosixPath = pathlib.WindowsPath


wound_model = load_learner(open("./trained_model/gpu_densenet169.pkl", "rb"), cpu = True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
labels = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8"]


classes_dict = {
    'Class_1': 'Abrasions',
    'Class_2': 'Bruises',
    'Class_3': 'Burns',
    'Class_4': 'Cut',
    'Class_5': 'Ingrown_nails',
    'Class_6': 'Laceration',
    'Class_7': 'Stab_wound'
}

def classify_img(uploaded_file, model : load_learner):
    is_wound, _, confidence = model.predict(uploaded_file)
    return classes_dict[is_wound], is_wound, confidence 

def get_class_info(class_data):
    info = {
        "name": class_data["name"],
        "description": class_data["description"],
        "symptoms": class_data["symptoms"],
        "first_aid": class_data["first_aid"]
    }
    return info

if uploaded_file is not None:
    predict_btn = st.button("Predict")

if uploaded_file:
        inputShape = (224, 224)
        
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        image = image.convert("RGB")
        image = image.resize(inputShape)

        class_name, wound_class, confidence = classify_img(image, wound_model)

        # Display prediction
        st.image(image, width=300)

        class_info = get_class_info(data[wound_class])
        st.subheader("Class Information")
        st.write(f"Predicted Class: {class_info['name']}")
        st.write(f"Description: {class_info['description']}")
        st.write("Symptoms:")
        for symptom in class_info['symptoms']:
            st.write(f"- {symptom}")
        st.write("First Aid:")
        for step in class_info['first_aid']:
            st.write(step)

if __name__ == "__main__":
        print("Start")

    