import streamlit as st
# from ocr_module import run_ocr   # Your separate OCR code

st.title("Handwritten OCR Converter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=300)

    if st.button("Convert to Text"):
        # Run OCR
        text = run_ocr(uploaded_file)
        st.subheader("OCR Output")
        st.write(text)