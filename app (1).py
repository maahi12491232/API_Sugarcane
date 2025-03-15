from inference_sdk import InferenceHTTPClient
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import base64
import tempfile

# Roboflow API settings
API_URL = "https://detect.roboflow.com"
API_KEY = "OcTWIdBRpv9vvKdqr7OQ"  
WORKSPACE_NAME = "project-jqwpc"  
WORKFLOW_ID = "custom-workflow-2"  

# Streamlit UI
st.title("Sugarcane Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((400, 300))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run inference
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

    # Creating a temporary file from uploaded image bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

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

            st.write(f"**Predicted Class:** {highest_confidence_class}")
            st.write(f"**Confidence:** {confidence:.2f}")

            # Display image with label
            fig, ax = plt.subplots(1, figsize=(6, 4))
            ax.imshow(image)
            label_text = f"{highest_confidence_class}: {confidence:.2f}"
            ax.text(10, 10, label_text, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            st.pyplot(fig)
            # Add CSS to style images
            st.markdown("""<style>img {
              max-width: 100%;
              height: auto;} 
              </style>""",unsafe_allow_html=True,)
        else:
            st.write("No predictions found.")
    else:
        st.write("No predictions found.")
