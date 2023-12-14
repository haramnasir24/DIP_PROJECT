import streamlit as st
from PIL import Image
import numpy as np
import cv2
import functions
import quantisation


def custom_title():
    st.markdown("""
        <h2 style='text-align: center; color: darkgreen; margin-top: 0px'>PHOTO EDITOR/PREPROCESSOR</h2>
    """, unsafe_allow_html=True)


def main():
    # st.title("PHOTO EDITOR/PREPROCESSOR")
    custom_title()

    with st.sidebar:

        # Allow the user to upload an image
        uploaded_image = st.file_uploader("Choose an image...", type=[
                                          "jpg", "jpeg", "png", "tiff"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)

        container = st.image(
            image, caption="Uploaded Image", use_column_width=False)

        # Get the size of the original image
        original_size = image.size
        st.write(
            f"Original Image Size: {original_size[0]} x {original_size[1]} pixels")


    with st.sidebar:

        st.header("Functions:")

        st.text("To Resize:")
        with st.form("resize_form"):
            width_input = st.number_input(
                "Enter the desired width:", value=0, step=1)
            height_input = st.number_input(
                "Enter the desired height:", value=0, step=1)

            # Add a submit button to the form
            submitted = st.form_submit_button("Resize")

            # Check if the form is submitted and both values are provided
            if submitted and width_input != 0 and height_input != 0:
                st.write(
                    f"Resizing to width: {width_input}, height: {height_input}")

                resized_image = functions.resize_image(
                    image, width_input, height_input)

                if resized_image is not None:
                    # Display the uploaded image
                    container.image(
                        resized_image, caption="Resized Image", use_column_width=False)

            elif submitted:
                st.warning(
                    "Please enter both width and height before resizing.")

        st.text("To Rotate:")

        with st.form("rotate_form"):

            angle = st.number_input(
                "Enter the desired angle:", value=0, step=1)

            uncropped_rotate = st.form_submit_button("Rotate but NOT Crop")
            cropped_rotate = st.form_submit_button("Rotate AND Crop")

            if (uncropped_rotate):

                uncropped_rotated_image = functions.rotate_not_cropped(
                    image, angle)

                if uncropped_rotated_image is not None:
                    # Display the uploaded image
                    container.image(
                        uncropped_rotated_image, caption="Rotated Image", use_column_width=False)

            if (cropped_rotate):

                cropped_rotated_image = functions.rotate_cropped(image, angle)

                if cropped_rotated_image is not None:
                    # Display the uploaded image
                    container.image(
                        cropped_rotated_image, caption="Rotated Image", use_column_width=False)

        st.text("Flip the image:")
        horizontal_flip = st.button("Horizontal Flip")

        if (horizontal_flip):

            horizontal_flipped_image = functions.horizontal_flip(image)

            if horizontal_flipped_image is not None:
                # Display the uploaded image
                container.image(horizontal_flipped_image,
                                caption="Rotated Image", use_column_width=False)

        Vertical_flip = st.button("Vertical Flip")

        if (Vertical_flip):

            vertical_flipped_image = functions.vertical_flip(image)

            if vertical_flipped_image is not None:
                # Display the uploaded image
                container.image(vertical_flipped_image,
                                caption="Rotated Image", use_column_width=False)

        # Add a slider
        contrast_input = st.slider("Change Contrast:", min_value=0,
                                   max_value=255, value=125, step=1, key='contrast_slider')

        # Check if the slider value has changed
        if st.button("Apply Contrast"):

            contrast_image = functions.linearContrastStretch(
                image, contrast_input)

            if contrast_image is not None:
                # Display the uploaded image
                container.image(
                    contrast_image, caption="Rotated Image", use_column_width=False)

        brightness_input = st.slider(
            "Change Brightness:", min_value=-255, max_value=255, value=0, step=1, key='brightness_slider')

        # Check if the slider value has changed
        if st.button("Apply Brightness"):

            bright_image = functions.image_brightness(image, brightness_input)

            if bright_image is not None:
                # Display the uploaded image
                container.image(
                    bright_image, caption="Rotated Image", use_column_width=False)

        quantisation_input = st.slider(
            "Quantise the image:", min_value=2, max_value=256, value=16, step=1, key='quantisation_slider')

        # Check if the slider value has changed
        if st.button("Apply Quantisation"):

            quantised_image = quantisation.quantize_image(
                image, quantisation_input)

            if quantised_image is not None:
                # Display the uploaded image
                container.image(
                    quantised_image, caption="Quantised Image", use_column_width=False)

            # Get the size of the quantized image
            quantised_size = quantised_image.size
            st.write(
                f"Quantised Image Size: {quantised_size[0]} x {quantised_size[1]} pixels")


if __name__ == "__main__":
    main()
