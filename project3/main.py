import cv2
import numpy as np 
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, 
    preprocess_input, 
    decode_predictions
)
from PIL import Image

def load_model():
    # cnn, able to look at images, weights = learned values that make the model work 
    # weights determined to = output to get a pretrained model that can classify videos 
    model = MobileNetV2(weights="imagenet")
    return model 

def preprocess_image(image):
    # convert image into array of diff numbers (rep pixels)
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img) # process so img ready to send img to mobile net 
    # takes single img to rep/look like mult images, "list of images" add another dim 
    img = np.expand_dims(img, axis=0) 
    return img

def classify_image(model, image):
    try: 
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        # convert img into string labels, get the top 3 decoded versions
        # model gets an image, gets array of values (%s), (0.9 = 90% its a dog), decode values to know 
        # the text label that is associated with it, grab top 3 with the highest percentages
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    
    except Exception as e: 
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Ai Image Classifier", page_icon="🖼️", layout="centered")
    st.title("AI Image Classifier")
    st.write("Uplaod an image and let AI tell you what's in it")

    # in streamlit, you can cache resrouces you use frequently  
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")

        if btn: 
            with st.spinner("Analyze Image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions: 
                    st.subheader("Predictions")
                    for _, label, score in predictions: 
                     # takes 2 dec places, show as % 
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()