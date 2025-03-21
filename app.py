import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
import tempfile

# Roboflow API settings
API_URL = "https://detect.roboflow.com"
API_KEY = "OcTWIdBRpv9vvKdqr7OQ"
WORKSPACE_NAME = "project-jqwpc"
WORKFLOW_ID = "custom-workflow"

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' height='100%' width='100%'%3E%3Ctext x='10' y='20' font-size='40' fill-opacity='0.25'%3E🍀%3C/text%3E%3C/svg%3E");
        background-repeat: round;
        background-size: 100px 100px;
        position: fixed; 
        margin: 0 auto; /* Center the app horizontally */
    }
    body {
        background-color: #f0f2f6; 
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .bottom-left {
        position: absolute;
        bottom: 0;
        left: 0;
        font-size: 40px;
    }
    .bottom-right {
        position: absolute;
        bottom: 0;
        right: 0;
        font-size: 40px;
    }    
    .image-container {
        border: 2px solid #ccc; /* Light gray border */
        border-radius: 10px;    /* Rounded corners */
        padding: 10px;         /* Padding inside the container */
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        margin-bottom: 20px;   /* Add some spacing below the image */
    }

    .image-container .stImage > img { /* Target images within Streamlit's image container */
        display: block;
        max-width: 100%;
        height: auto;
        border-radius: 8px;  /* Slightly rounded image corners */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use markdown to add emojis to all four corners (within the CSS)
st.markdown(
    """
    <div class="bottom-left">🍃 🍁 🍃 🍁 🍃 🍁</div>
    <div class="bottom-right">🍁 🍃 🍁 🍃 🍁 🍃</div>
    """,
    unsafe_allow_html=True
)
# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "login"  # Possible values: "login", "prediction"
if 'login_error' not in st.session_state:
    st.session_state.login_error = False

# Callback functions
def login_callback():
    # Get the current username and password from session state keys used in text_input
    username = st.session_state.username_input
    password = st.session_state.password_input
    # Replace this logic with your actual validation mechanism
    if username == "admin" and password == "password":
        st.session_state.page = "prediction"
        st.session_state.login_error = False
    else:
        st.session_state.login_error = True

def logout_callback():
    st.session_state.page = "login"
    # Optionally clear username/password if desired:
    st.session_state.username_input = ""
    st.session_state.password_input = ""

# UI Sections
def login_section():
    st.markdown('''
    <h1 style="text-align: center;
               font-family: 'Times New Roman', serif;
               font-size: 45px;
               font-weight: bold;
               text-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
        🌿Sugarcane Disease Detection
    </h1>
''', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; font-size: 120px;">🧑🏻‍🌾</h1>', unsafe_allow_html=True)
    st.markdown('''
    <h3 style="text-align: center;
               font-family: 'Times New Roman', serif;
               font-size: 40px;
               font-weight: bold;
               text-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
        Login to access the application🌿
    </h3>
''', unsafe_allow_html=True)
    st.text_input("Username", key="username_input")
    st.text_input("Password", type="password", key="password_input")

    # The login button uses the callback so that one click is enough.
    st.button("Login", key="login_button", on_click=login_callback)

    if st.session_state.login_error:
        st.error("Incorrect username or password.")

def prediction_section():
    # Sidebar: Logout button using callback
    st.sidebar.button("Logout", key="logout_button", on_click=logout_callback)
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    #model_path = st.sidebar.text_input("Enter the path to the trained model", "/content/drive/MyDrive/Colab Notebooks/Models keras/efficientnetB0_model.keras")

    #@st.cache_resource
    #def load_model(model_path):
        #return tf.keras.models.load_model(model_path)

    #@st.cache_data
    #def load_and_preprocess_image(uploaded_image):
      #image = Image.open(uploaded_image)
      #image = image.resize((224, 224))
      #img_array = np.array(image) / 255.0  # Normalize
      #return image, img_array # Return both image and image array

    disease_info = {
        "Rust": {
                "Cause": "Fungus called Puccinia melanocephala. Thrives in warm, humid conditions.",
                "Symptoms": "Yellowish-brown pustules on leaves, stunted growth.",
                "Suggestions": "Ensure proper crop spacing, avoid over-irrigation, consider resistant varieties or fungicides."
                },
        "RedRot": {
                "Cause": "Fungal disease caused by Colletotrichum falcatum. Often due to waterlogging or infected material.",
                "Symptoms": "Reddish discoloration of internodes, white spots, sour smell.",
                "Suggestions": "Use disease-free planting material, improve drainage, apply fungicides."
                },
        "Mosaic": {
                "Cause": "Viral infection, often transmitted by aphids or contaminated tools.",
                "Symptoms": "Irregular patches of light and dark green on leaves, reduced photosynthesis.",
                "Suggestions": "Control aphids, sterilize tools, use resistant varieties."
                },
        "Healthy": {
                "Cause": "No disease detected.",
                "Symptoms": "Leaves are healthy with no visible issues.",
                "Suggestions": "Maintain proper care for the plant."
                },
        "Yellow": {
                "Cause": "Potential nutrient deficiency or other minor stress factors.",
                "Symptoms": "Yellowish discoloration of leaves.",
                "Suggestions": "Check soil nutrients and improve fertilization practices."
                }
    }
    st.markdown("<h1 style='text-align: center;'>🌿Sugarcane Disease Detection</h1>", unsafe_allow_html=True)
    uploaded_images = st.file_uploader("Choose sugar cane leaf images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)  # Initialize client
    #if 'model' not in st.session_state:
        #st.session_state.model = load_model(model_path)
    #model = st.session_state.model

    if uploaded_images: #and model_path:
        #class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
        for uploaded_image in uploaded_images:
            image = Image.open(uploaded_image)
            image = image.resize((400, 300))  # Resize as needed
            #image, img_array = load_and_preprocess_image(uploaded_image)
            #img_array = img_array.reshape(1, 224, 224, 3)
            #with st.spinner('Classifying...'):
                #predictions = model.predict(img_array)
            #predicted_class = class_names[np.argmax(predictions, axis=1)[0]]
            #confidence = np.max(predictions, axis=1)[0]

            # Creating a temporary file from uploaded image bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_image.getvalue())
                temp_file_path = temp_file.name

            # Run inference using Roboflow API
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": temp_file_path},
                use_cache=True
            )

            # Display results
            if result and result[0].get("predictions"):
                predictions = result[0]["predictions"]
                predicted_classes = predictions.get("predicted_classes", [])
                class_confidences = predictions.get("predictions", {})

                if predicted_classes:
                    highest_confidence_class = predicted_classes[0]
                    confidence = class_confidences.get(highest_confidence_class, {}).get("confidence")

                    if confidence >= threshold: #Apply Threshold
                        # Display image and information
                        with st.container():
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            fig, ax = plt.subplots(1, figsize=(6, 4))
                            ax.imshow(image)
                            label_text = f"{highest_confidence_class}: {confidence:.2f}"
                            ax.text(10, 10, label_text, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                            st.pyplot(fig)
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.write(f"Prediction: {highest_confidence_class} (Confidence: {confidence:.2f})")

                        with st.expander("Disease Information"):
                            info = disease_info.get(highest_confidence_class, {})
                            for key, value in info.items():
                                st.write(f"**{key}:** {value}")
                    else:
                        st.write(f"Prediction confidence ({confidence:.2f}) below threshold ({threshold:.2f}), not displaying.") 
                else:
                    st.write("No predictions found.")
            else:
                st.write("No predictions found.")


# Main App Flow
if st.session_state.page == "login":
    login_section()
elif st.session_state.page == "prediction":
    prediction_section()
