import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure the page
st.set_page_config(
    page_title="DiseaseCapt",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling with contrasting colors
st.markdown("""
    <style>
        .stApp {
            background-color: #002B5B;
            color: #F5C518;
        }
        .stButton > button {
            background-color: #F5C518;
            color: #002B5B !important;
            border-radius: 4px;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1rem;
        }
        .css-1d391kg {
            background-color: #001F3F;
        }
        h2, h3, h1 {
            color: #F5C518 !important;
            font-weight: bold !important;
            text-align: center;
        }
        p, li {
            color: #ffffff !important;
            font-size: 16px !important;
            text-align: center;
        }
        a {
            color: #F5C518 !important;
            font-weight: 500;
        }
        .sidebar-text {
            color: #F5C518 !important;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("🌱 DiseaseCapt")
st.sidebar.markdown("<p class='sidebar-text'>Plant Disease Detection</p>", unsafe_allow_html=True)
app_mode = st.sidebar.selectbox("Select page", ["home", "disease recognition"])

# Display plant icons
st.sidebar.markdown("🌱🌿🍀🌵🌳")
st.sidebar.markdown("""
    DiseaseCapt helps detect plant diseases accurately.
    Upload an image to get started!
""")

if app_mode == "home":
    st.markdown("<h1>DiseaseCapt: Plant Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
        <h3>Welcome to DiseaseCapt!</h3>
        <p>
            DiseaseCapt is an AI-powered tool designed to detect plant diseases with precision.
            Simply upload an image of a plant leaf, and our system will analyze and provide a diagnosis.
        </p>
        <h3>Tools & Libraries Used:</h3>
        <p>
            - Streamlit<br>
            - TensorFlow<br>
            - NumPy<br>
            - PIL (Pillow)<br>
            - Keras
        </p>
    """, unsafe_allow_html=True)
    st.markdown("🌱🌿🍀🌵🌳", unsafe_allow_html=True)

elif app_mode == "disease recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            class_names = ['Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
                           'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy', 'Corn - Gray Leaf Spot',
                           'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy', 'Grape - Black Rot',
                           'Grape - Black Measles', 'Grape - Leaf Blight', 'Grape - Healthy', 'Orange - Citrus Greening',
                           'Peach - Bacterial Spot', 'Peach - Healthy', 'Pepper - Bacterial Spot', 'Pepper - Healthy',
                           'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy',
                           'Soybean - Healthy', 'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
                           'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold',
                           'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot',
                           'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy']
            st.success(f"Model is Predicting it's a {class_names[result_index]}")

# Footer
st.markdown("<hr style='margin: 30px 0px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #F5C518;'>DiseaseCapt © 2025 | Empowering Sustainable Agriculture</p>", unsafe_allow_html=True)
