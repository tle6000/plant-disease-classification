import pathlib
import platform

# Fix for cross-platform compatibility
if platform.system() != 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath = pathlib.Path

import tempfile
import os
from pathlib import Path
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import torch

# Additional fix for PyTorch classes issue
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

@st.cache_resource
def load_model(path):
    try:
        # Check if CUDA is available, otherwise use CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Try loading without cpu flag first (allows GPU usage)
        try:
            learner = load_learner(path)
            st.info(f"Model loaded successfully on {device.upper()}")
            return learner
        except:
            # Fallback to CPU if GPU loading fails
            learner = load_learner(path, cpu=True)
            st.info("Model loaded successfully on CPU (fallback)")
            return learner
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def safe_predict(learner, img_path):
    """Safely predict with error handling"""
    try:
        pred_class, pred_idx, outputs = learner.predict(img_path)
        
        # Debug: Show what we got
        st.write(f"Debug - pred_class type: {type(pred_class)}")
        st.write(f"Debug - pred_class value: {pred_class}")
        
        # Handle different types of pred_class
        if isinstance(pred_class, dict):
            # If it's a dict, try common keys
            if 'label' in pred_class:
                label = str(pred_class['label'])
            elif 'class' in pred_class:
                label = str(pred_class['class'])
            elif 'prediction' in pred_class:
                label = str(pred_class['prediction'])
            else:
                # Use the first value or convert whole dict to string
                label = str(list(pred_class.values())[0]) if pred_class else str(pred_class)
        elif hasattr(pred_class, 'item'):
            # If it's a tensor, get the item
            label = str(pred_class.item())
        elif hasattr(pred_class, '__str__'):
            # If it has a string representation
            label = str(pred_class)
        else:
            # Last resort - convert to string
            label = repr(pred_class)
            
        confidence = float(outputs[pred_idx])
        return label, confidence, True
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write(f"Error type: {type(e)}")
        return None, None, False

def main():
    st.set_page_config(page_title="Plant Disease Classification", layout="centered")
    st.title("Plant Disease Classification App")
    st.write("Upload a plant leaf image or take a photo and we will predict the disease.")

    # Load model with error handling
    learner = load_model("final_model_resnet50.pkl")
    
    if learner is None:
        st.error("Failed to load the model. Please check if the model file exists and is compatible.")
        return

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

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            temp_path = tf.name
            
        try:
            # Save image to temporary file
            img.save(temp_path)
            
            # Make prediction
            label, confidence, success = safe_predict(learner, temp_path)
            
            if success:
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {confidence:.2%}")
            else:
                st.error("Failed to make prediction. Please try again with a different image.")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    main()