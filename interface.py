import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the trained model
loaded_model = tf.keras.models.load_model('Pneumoniadetection.h5', compile=False)

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

# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torchvision import models

# # Load the trained model
# loaded_model = tf.keras.models.load_model('Pneumoniadetection.h5', compile=False)

# # Define class labels
# classes = ['NORMAL', 'PNEUMONIA']

# # Function to check if image is chest x-ray
# # def is_chest_xray(image):
# #     # Preprocess the image
# #     preprocess = transforms.Compose([
# #         transforms.Resize(256),
# #         transforms.CenterCrop(224),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #     ])
# #     img_tensor = preprocess(image)
# #     img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

# #     # Load a pre-trained ResNet model
# #     model = models.resnet18(pretrained=True)
# #     model.eval()

# #     # Make prediction
# #     with torch.no_grad():
# #         outputs = model(img_tensor)

# #     # Get predicted label
# #     _, predicted = outputs.max(1)
# #     class_index = predicted.item()
    
# #     # Replace this list with actual class labels in your dataset
# #     class_labels = ['not_xray', 'xray']

# #     # Check if the predicted label corresponds to a chest x-ray
# #     if class_labels[class_index] == 'xray':
# #         return True
# #     else:
# #         return False

# # import torch

# def is_chest_xray(image):
#     # Convert PIL.Image to PyTorch tensor
#     img_tensor = transforms.ToTensor()(image)
    
#     # If the image has only one channel, convert it to three channels (grayscale to RGB)
#     if img_tensor.shape[0] == 1:
#         img_tensor = torch.cat((img_tensor, img_tensor, img_tensor), dim=0)
    
#     # Preprocess the image
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         # No need for normalization here since ToTensor already does that
#     ])
#     img_tensor = preprocess(img_tensor)  # Apply preprocessing
    
#     # Add batch dimension
#     img_tensor = img_tensor.unsqueeze(0)

#     # Load a pre-trained ResNet model
#     model = models.resnet18(pretrained=True)
#     model.eval()

#     # Make prediction
#     with torch.no_grad():
#         outputs = model(img_tensor)

#     # Get predicted label
#     _, predicted = outputs.max(1)
#     class_index = predicted.item()
    
#     # Replace this list with actual class labels in your dataset
#     class_labels = ['not_xray', 'xray']

#     # Check if the predicted label corresponds to a chest x-ray
#     if class_labels[class_index] == 'xray':
#         return True
#     else:
#         return False


# # Define function for making predictions
# def predict(image):
#     # Preprocess the image
#     img = image.resize((224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
    
#     # Make predictions
#     predictions = loaded_model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#     class_index = tf.argmax(score).numpy()
#     class_label = classes[class_index]
#     confidence = score[class_index].numpy()
    
#     return class_label, confidence

# # Image classification page
# def classify_image():
#     st.title('Pneumonia Detection App')
#     st.sidebar.title('Upload Image')
    
#     # Upload image
#     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#         st.write("")
        
#         # Check if the uploaded image is a chest x-ray
#         if is_chest_xray(image):
#             # Predict button
#             if st.button('Predict'):
#                 st.write("Classifying...")
                
#                 # Make prediction
#                 class_label, confidence = predict(image)
                
#                 st.write(f'Prediction: {class_label}')
#                 st.write(f'Confidence: {confidence * 100:.2f}%')
#         else:
#             st.write("Please upload a chest X-ray image.")

# # Streamlit app
# def main():
#     st.set_page_config(page_title="Pneumonia Detection App", page_icon=":lungs:")
#     page = st.sidebar.selectbox("Navigate", ["Classify Image"])

#     if page == "Classify Image":
#         classify_image()

# if __name__ == '__main__':
#     main()



# # # import streamlit as st
# # # import tensorflow as tf
# # # import numpy as np
# # # from PIL import Image
# # # from skimage.metrics import structural_similarity as ssim
# # # from skimage.transform import resize

# # # # Load the reference chest X-ray image
# # # # reference_image_path = r'C:\Users\Ataf Mirza\Desktop\CV\X-ray.jpeg'  # Path to your reference chest X-ray image
# # # # reference_image = Image.open(reference_image_path)

# # # reference_image_path = 'C:/Users/Ataf Mirza/Desktop/CV/X-ray.jpeg'

# # # # Load the reference chest X-ray image
# # # reference_image = Image.open(reference_image_path)

# # # # Load the trained model
# # # loaded_model = tf.keras.models.load_model('Pneumoniadetection.h5', compile=False)

# # # # Define class labels
# # # classes = ['NORMAL', 'PNEUMONIA']

# # # # Define function for making predictions
# # # def predict(image):
# # #     # Preprocess the image
# # #     img = image.resize((224, 224))
# # #     img_array = tf.keras.preprocessing.image.img_to_array(img)
# # #     img_array = tf.expand_dims(img_array, 0)
    
# # #     # Make predictions
# # #     predictions = loaded_model.predict(img_array)
# # #     score = tf.nn.softmax(predictions[0])
# # #     class_index = tf.argmax(score).numpy()
# # #     class_label = classes[class_index]
# # #     confidence = score[class_index].numpy()
    
# # #     return class_label, confidence

# # # # Check if the uploaded image is similar to the reference chest X-ray image
# # # # def is_chest_xray(image):
# # # #     # Calculate the Structural Similarity Index (SSIM) between the uploaded image and the reference image
# # # #     similarity_index = ssim(image, reference_image, multichannel=True)
    
# # # #     # Set a threshold for similarity
# # # #     similarity_threshold = 0.8  # Adjust this threshold as needed
    
# # # #     # If the similarity index is above the threshold, consider it a chest X-ray
# # # #     if similarity_index > similarity_threshold:
# # # #         return True
# # # #     else:
# # # #         return False


# # # # import numpy as np
# # # # from skimage.metrics import structural_similarity as ssim

# # # # Check if the uploaded image is similar to the reference chest X-ray image
# # # # def is_chest_xray(image):
# # # #     # Convert PIL Image objects to NumPy arrays
# # # #     uploaded_image_array = np.array(image)
# # # #     reference_image_array = np.array(reference_image)
    
# # # #     # Calculate the Structural Similarity Index (SSIM) between the uploaded image and the reference image
# # # #     similarity_index = ssim(uploaded_image_array, reference_image_array, multichannel=True)
    
# # # #     # Set a threshold for similarity
# # # #     similarity_threshold = 0.8  # Adjust this threshold as needed
    
# # # #     # If the similarity index is above the threshold, consider it a chest X-ray
# # # #     if similarity_index > similarity_threshold:
# # # #         return True
# # # #     else:
# # # #         return False

# # # # import numpy as np
# # # # from skimage.metrics import structural_similarity as ssim
# # # # from skimage.transform import resize

# # # # # Check if the uploaded image is similar to the reference chest X-ray image
# # # # def is_chest_xray(image):
# # # #     # Convert PIL Image objects to NumPy arrays
# # # #     uploaded_image_array = np.array(image)
# # # #     reference_image_array = np.array(reference_image)
    
# # # #     # Resize both images to a standard size (e.g., 224x224)
# # # #     standard_size = (224, 224)
# # # #     uploaded_image_resized = resize(uploaded_image_array, standard_size, anti_aliasing=True)
# # # #     reference_image_resized = resize(reference_image_array, standard_size, anti_aliasing=True)
    
# # # #     # Calculate the Structural Similarity Index (SSIM) between the resized images
# # # #     similarity_index = ssim(uploaded_image_resized, reference_image_resized, multichannel=True)
    
# # # #     # Set a threshold for similarity
# # # #     similarity_threshold = 0.8  # Adjust this threshold as needed
    
# # # #     # If the similarity index is above the threshold, consider it a chest X-ray
# # # #     if similarity_index > similarity_threshold:
# # # #         return True
# # # #     else:
# # # #         return False\

# # # # def is_chest_xray(image):
# # # #     # Convert PIL Image objects to NumPy arrays
# # # #     uploaded_image_array = np.array(image)
# # # #     reference_image_array = np.array(reference_image)
    
# # # #     # Resize both images to a standard size (e.g., 224x224)
# # # #     standard_size = (224, 224)
# # # #     uploaded_image_resized = resize(uploaded_image_array, standard_size, anti_aliasing=True)
# # # #     reference_image_resized = resize(reference_image_array, standard_size, anti_aliasing=True)
    
# # # #     # Calculate the Structural Similarity Index (SSIM) between the resized images
# # # #     similarity_index = ssim(uploaded_image_resized, reference_image_resized, multichannel=True, data_range=1.0)
    
# # # #     # Set a threshold for similarity
# # # #     similarity_threshold = 0.5  # Adjust this threshold as needed
    
# # # #     # If the similarity index is above the threshold, consider it a chest X-ray
# # # #     if similarity_index > similarity_threshold:
# # # #         return True
# # # #     else:
# # # # #         return False
# # # # import numpy as np
# # # # from skimage.metrics import structural_similarity as ssim
# # # # from skimage.transform import resize

# # # # Check if the uploaded image is similar to the reference chest X-ray image
# # # def is_chest_xray(uploaded_image, reference_image):
# # #     # Convert PIL Image objects to NumPy arrays
# # #     uploaded_image_array = np.array(uploaded_image)
# # #     reference_image_array = np.array(reference_image)
    
# # #     # Resize both images to a standard size (e.g., 224x224)
# # #     standard_size = (224, 224)
# # #     uploaded_image_resized = resize(uploaded_image_array, standard_size, anti_aliasing=True)
# # #     reference_image_resized = resize(reference_image_array, standard_size, anti_aliasing=True)
    
# # #     # Calculate the Structural Similarity Index (SSIM) between the resized images
# # #     similarity_index = ssim(uploaded_image_resized, reference_image_resized, multichannel=True, data_range=1.0)
    
# # #     # Set a threshold for similarity
# # #     similarity_threshold = 0.8  # Adjust this threshold as needed
    
# # #     # If the similarity index is above the threshold, consider it a chest X-ray
# # #     if similarity_index > similarity_threshold:
# # #         return True
# # #     else:
# # #         return False


# # # # Home page
# # # def home():
# # #     st.title('Pneumonia Detection App')
# # #     st.write("""
# # #     This app is designed to classify whether an uploaded image contains pneumonia or is normal, but only if it is similar to a chest X-ray.
# # #     """)
# # #     st.write("## Instructions:")
# # #     st.write("1. Upload an image using the sidebar on the left.")
# # #     st.write("2. Click on the 'Predict' button to classify the image.")
# # #     st.write("3. The app will display the prediction along with confidence, but only if the image is similar to a chest X-ray.")

# # # # Image classification page
# # # # def classify_image():
# # # #     st.title('Pneumonia Detection App')
# # # #     st.sidebar.title('Upload Image')
    
# # # #     # Upload image
# # # #     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
# # # #     if uploaded_file is not None:
# # # #         image = Image.open(uploaded_file)
        
# # # #         # Check if the uploaded image is similar to a chest X-ray
# # # #         if is_chest_xray(image):
# # # #             st.image(image, caption='Uploaded Image', use_column_width=True)
# # # #             st.write("")
            
# # # #             # Predict button
# # # #             if st.button('Predict'):
# # # #                 st.write("Classifying...")
                
# # # #                 # Make prediction
# # # #                 class_label, confidence = predict(image)
                
# # # #                 st.write(f'Prediction: {class_label}')
# # # #                 st.write(f'Confidence: {confidence * 100:.2f}%')
# # # #         else:
# # # #             st.write("The uploaded image does not appear to be similar to a chest X-ray. Please upload an image similar to a chest X-ray.")

# # # def classify_image():
# # #     st.title('Pneumonia Detection App')
# # #     st.sidebar.title('Upload Image')
    
# # #     # Upload image
# # #     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
# # #     if uploaded_file is not None:
# # #         image = Image.open(uploaded_file)
        
# # #         # Load the reference chest X-ray image
# # #         reference_image_path = 'C:/Users/Ataf Mirza/Desktop/CV/X-ray.jpeg'
# # #         reference_image = Image.open(reference_image_path)
        
# # #         # Check if the uploaded image is similar to the reference chest X-ray image
# # #         if is_chest_xray(image, reference_image):  # Provide both uploaded image and reference image
# # #             st.image(image, caption='Uploaded Image', use_column_width=True)
# # #             st.write("")
            
# # #             # Predict button
# # #             if st.button('Predict'):
# # #                 st.write("Classifying...")
                
# # #                 # Make prediction
# # #                 class_label, confidence = predict(image)
                
# # #                 st.write(f'Prediction: {class_label}')
# # #                 st.write(f'Confidence: {confidence * 100:.2f}%')
# # #         else:
# # #             st.write("The uploaded image does not appear to be similar to a chest X-ray. Please upload an image similar to a chest X-ray.")

# # # # Streamlit app
# # # def main():
# # #     st.set_page_config(page_title="Pneumonia Detection App", page_icon=":lungs:")
# # #     page = st.sidebar.selectbox("Navigate", ["Home", "Classify Image"])
    
# # #     if page == "Home":
# # #         home()
# # #     elif page == "Classify Image":
# # #         classify_image()

# # # if __name__ == '__main__':
# # #     main()


# # import streamlit as st
# # import tensorflow as tf
# # import numpy as np
# # from PIL import Image
# # from skimage.metrics import structural_similarity as ssim
# # from skimage.transform import resize

# # # Set page configuration
# # st.set_page_config(page_title="Pneumonia Detection App", page_icon=":lungs:")

# # # Load the trained model
# # @st.cache(allow_output_mutation=True)
# # def load_model():
# #     model = tf.keras.models.load_model('Pneumoniadetection.h5', compile=False)
# #     return model

# # model = load_model()

# # # Define class labels
# # classes = ['NORMAL', 'PNEUMONIA']

# # # Define function for making predictions
# # def predict(image):
# #     # Preprocess the image
# #     img = image.resize((224, 224))
# #     img_array = tf.keras.preprocessing.image.img_to_array(img)
# #     img_array = tf.expand_dims(img_array, 0)
    
# #     # Make predictions
# #     predictions = model.predict(img_array)
# #     score = tf.nn.softmax(predictions[0])
# #     class_index = tf.argmax(score).numpy()
# #     class_label = classes[class_index]
# #     confidence = score[class_index].numpy()
    
# #     return class_label, confidence

# # # Check if the uploaded image is similar to the reference chest X-ray image
# # def is_chest_xray(uploaded_image, reference_image):
# #     # Convert PIL Image objects to NumPy arrays
# #     uploaded_image_array = np.array(uploaded_image)
# #     reference_image_array = np.array(reference_image)
    
# #     # Resize both images to a standard size (e.g., 224x224)
# #     standard_size = (224, 224)
# #     uploaded_image_resized = resize(uploaded_image_array, standard_size, anti_aliasing=True)
# #     reference_image_resized = resize(reference_image_array, standard_size, anti_aliasing=True)
    
# #     # Calculate the Structural Similarity Index (SSIM) between the resized images
# #     similarity_index = ssim(uploaded_image_resized, reference_image_resized, multichannel=True, data_range=1.0)
    
# #     # Set a threshold for similarity
# #     similarity_threshold = 0.3  # Adjust this threshold as needed
    
# #     # If the similarity index is above the threshold, consider it a chest X-ray
# #     if similarity_index > similarity_threshold:
# #         return True
# #     else:
# #         return False

# # # Home page
# # def home():
# #     st.title('Pneumonia Detection App')
# #     st.write("""
# #     This app is designed to classify whether an uploaded image contains pneumonia or is normal, but only if it is similar to a chest X-ray.
# #     """)
# #     st.write("## Instructions:")
# #     st.write("1. Upload an image using the sidebar on the left.")
# #     st.write("2. Click on the 'Predict' button to classify the image.")
# #     st.write("3. The app will display the prediction along with confidence, but only if the image is similar to a chest X-ray.")

# # # Image classification page
# # def classify_image():
# #     st.title('Pneumonia Detection App')
# #     st.sidebar.title('Upload Image')
    
# #     # Upload image
# #     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
# #     if uploaded_file is not None:
# #         image = Image.open(uploaded_file)
        
# #         # Load the reference chest X-ray image
# #         reference_image_path = 'C:/Users/Ataf Mirza/Desktop/CV/X-ray.jpeg'
# #         reference_image = Image.open(reference_image_path)
        
# #         # Check if the uploaded image is similar to the reference chest X-ray image
# #         if is_chest_xray(image, reference_image):  # Provide both uploaded image and reference image
# #             st.image(image, caption='Uploaded Image', use_column_width=True)
# #             st.write("")
            
# #             # Predict button
# #             if st.button('Predict'):
# #                 st.write("Classifying...")
                
# #                 # Make prediction
# #                 class_label, confidence = predict(image)
                
# #                 st.write(f'Prediction: {class_label}')
# #                 st.write(f'Confidence: {confidence * 100:.2f}%')
# #         else:
# #             st.write("The uploaded image does not appear to be similar to a chest X-ray. Please upload an image similar to a chest X-ray.")

# # # Streamlit app
# # def main():
# #     page = st.sidebar.selectbox("Navigate", ["Home", "Classify Image"])
    
# #     if page == "Home":
# #         home()
# #     elif page == "Classify Image":
# #         classify_image()

# # if __name__ == '__main__':
# #     main()






