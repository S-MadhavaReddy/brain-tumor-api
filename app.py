import streamlit as st
import requests

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    layout="centered"
)

st.title("Brain Tumor Classification")

# -------------------------
# API URL
# -------------------------
API_URL = "http://127.0.0.1:8000/predict"

# -------------------------
# CLASS NAMES
# -------------------------
class_names = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file is not None:

    with st.spinner("Predicting..."):

        files = {
            "file": uploaded_file.getvalue()
        }

        try:

            response = requests.post(
                API_URL,
                files=files,
                timeout=120
            )

            # -------------------------
            # API ERROR
            # -------------------------
            if response.status_code != 200:

                st.error("API Error")

                st.write(response.text)

            else:

                result = response.json()

                # -------------------------
                # RESULTS
                # -------------------------
                st.header("Prediction")

                st.markdown(
                    f"## Class: {result['prediction']}"
                )

                st.markdown(
                    f"### Confidence: {result['confidence']:.2f}%"
                )

                # -------------------------
                # PROBABILITIES
                # -------------------------
                st.subheader("Class Probabilities")

                for i, cls in enumerate(class_names):

                    prob = result["probabilities"][i]

                    st.write(f"{cls}: {prob:.2f}%")

                    st.progress(prob / 100)

                # -------------------------
                # ORIGINAL IMAGE
                # -------------------------
                st.markdown("---")

                st.subheader("Uploaded MRI")

                st.image(
                    uploaded_file,
                    width=350
                )

        except Exception as e:

            st.error(str(e))