import pathlib
pathlib.PosixPath = pathlib.Path


import streamlit as st
from fastai.vision.all import load_learner
from PIL import Image

@st.cache_resource
def load_model(path):
    model = load_learner(path)
    return model

def main():
    st.set_page_config(page_title="Plant Disease Classification", layout="centered")
    st.title("Plant Disease Classification App")
    st.write("Upload a plant leaf image or take a photo and we will predict the disease!!!!!.")

    learner = load_model("final_model_resnet50.pkl")  


    upload = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    camera = st.camera_input("Or take a photo")

    img = None
    if upload:
        img = Image.open(upload).convert('RGB')
    elif camera:
        img = Image.open(camera).convert('RGB')

    if img:
        st.image(img, caption='Input Image', use_container_width=True)
        st.write("Identifying disease...")

        pred_class, pred_idx, outputs = learner.predict(img)
        label = str(pred_class)
        confidence = float(outputs[pred_idx])

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2%}")


main()
