import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the trained model
model = r"C:\Users\Ataf Mirza\Desktop\computer vison\Pneumonia-Detection-App\Pneumoniadetection.h5"
loaded_model = tf.keras.models.load_model(model, compile=False)

# Define class labels
classes = ['NORMAL', 'PNEUMONIA']

# Define function for making predictions
def predict(image):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Make predictions
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_index = tf.argmax(score).numpy()
    class_label = classes[class_index]
    confidence = score[class_index].numpy()
    
    return class_label, confidence

# Home page
def home():
    st.title('Pneumonia Detection App')
    st.write("""
    This app is designed to classify whether an uploaded chest X-ray image contains pneumonia or is normal.
    """)
    st.write("## Instructions:")
    st.write("1. Upload a chest X-ray image using the sidebar on the left.")
    st.write("2. Click on the 'Predict' button to classify the image.")
    st.write("3. The app will display the prediction along with confidence.")

# Image classification page
def classify_image():
    st.title('Pneumonia Detection App')
    st.sidebar.title('Upload Image')
    
    # Upload image
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        
        # Predict button
        if st.button('Predict'):
            st.write("Classifying...")
            
            # Make prediction
            class_label, confidence = predict(image)
            
            st.write(f'Prediction: {class_label}')
            st.write(f'Confidence: {confidence * 100:.2f}%')

# Streamlit app
def main():
    st.set_page_config(page_title="Pneumonia Detection App", page_icon=":lungs:")
    page = st.sidebar.selectbox("Navigate", ["Home", "Classify Image"])
    
    if page == "Home":
        home()
    elif page == "Classify Image":
        classify_image()

if __name__ == '__main__':
    main()






