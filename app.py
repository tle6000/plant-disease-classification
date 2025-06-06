import pathlib
import platform

if platform.system() != 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath = pathlib.Path

import tempfile
import os
from pathlib import Path
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import torch


import warnings
warnings.filterwarnings('ignore', category=UserWarning)

@st.cache_resource
def load_model(path):
    try:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            learner = load_learner(path)
            st.info(f"Model loaded successfully on {device.upper()}")
            return learner
        except:

            learner = load_learner(path, cpu=True)
            st.info("Model loaded successfully on CPU (fallback)")
            return learner
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def safe_predict(learner, img_path):
    try:

        try:
            from PIL import Image as PILImage_base
            from fastai.vision.all import PILImage, tensor
            import torch
            from torchvision import transforms
            

            pil_img = PILImage_base.open(img_path).convert('RGB')
            

            transform = transforms.Compose([
                transforms.Resize((256, 256)),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  
            ])
            
            img_tensor = transform(pil_img).unsqueeze(0)  

            model = learner.model
            model.eval()
            
            with torch.no_grad():
                outputs = model(img_tensor)
                if hasattr(outputs, 'softmax'):
                    probs = outputs.softmax(dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)
                
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0][pred_idx].item()
                

                if hasattr(learner, 'dls') and hasattr(learner.dls, 'vocab'):
                    class_names = learner.dls.vocab
                    label = str(class_names[pred_idx])
                elif hasattr(learner, 'data') and hasattr(learner.data, 'classes'):
                    class_names = learner.data.classes
                    label = str(class_names[pred_idx])
                else:

                    label = f"Class_{pred_idx}"
                
                return label, confidence, True
                
        except Exception as manual_error:
            st.write(f"Manual prediction failed: {manual_error}")
            
           
            try:
                pred_class, pred_idx, outputs = learner.predict(img_path)
                
                if isinstance(pred_class, dict):
                    label = str(list(pred_class.values())[0]) if pred_class else "Unknown"
                else:
                    label = str(pred_class)
                    
                confidence = float(outputs[pred_idx])
                return label, confidence, True
                
            except:
                raise Exception("Both manual and FastAI prediction methods failed")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, False

def main():
    st.set_page_config(page_title="Plant Disease Classification", layout="centered")
    st.title("Plant Disease Classification App")
    st.write("Upload a plant leaf image or take a photo and we will predict the disease.")


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


        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            temp_path = tf.name
            
        try:

            img.save(temp_path)
            

            label, confidence, success = safe_predict(learner, temp_path)
            
            if success:
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {confidence:.2%}")
            else:
                st.error("Failed to make prediction. Please try again with a different image.")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            try:
                os.remove(temp_path)
            except:
                pass


main()