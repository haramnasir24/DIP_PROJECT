import streamlit as st
from PIL import Image
import numpy as np
import cv2
import functions
import intensity_transform_laws
import histogramEqualisation
import quantisation
import otsu
import waterMark

# Declare image variable outside the main function
image = None


def custom_title():
    st.markdown("""
        <h2 style='text-align: center; color: white; margin-top: 0px'>PHOTO EDITOR/PREPROCESSOR</h2>
    """, unsafe_allow_html=True)


def main():
    global image
    # st.title("PHOTO EDITOR/PREPROCESSOR")
    custom_title()

    with st.sidebar:

        # Allow the user to upload an image
        uploaded_image = st.file_uploader("Choose an image...", type=[
                                          "jpg", "jpeg", "png", "tiff"])

        # image = cv2.imread(uploaded_image)

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)

        width, height = image.size

        if (width > 800 or height > 500):

            container = st.image(
                image, caption="Uploaded Image", use_column_width=True)

        else:

            container = st.image(
                image, caption="Uploaded Image", use_column_width=False)

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

            uncropped_rotate = st.form_submit_button("Rotate without crop")
            cropped_rotate = st.form_submit_button("Rotate and Crop")

            if (uncropped_rotate):

                uncropped_rotated_image = functions.rotate_not_cropped(
                    image, angle)

                if uncropped_rotated_image is not None:
                    # Display the uploaded image
                    container.image(
                        uncropped_rotated_image, caption="Cropped Image", use_column_width=False)

            if (cropped_rotate):

                cropped_rotated_image = functions.rotate_cropped(image, angle)

                if cropped_rotated_image is not None:
                    # Display the uploaded image
                    container.image(
                        cropped_rotated_image, caption="Cropped Image", use_column_width=False)

        st.text("Flip the image:")
        horizontal_flip = st.button("Horizontal Flip")

        if (horizontal_flip):

            horizontal_flipped_image = functions.horizontal_flip(image)

            if horizontal_flipped_image is not None:
                # Display the uploaded image
                container.image(horizontal_flipped_image,
                                caption="Flipped Image", use_column_width=False)

        vertical_flip = st.button("Vertical Flip")

        if (vertical_flip):

            vertical_flipped_image = functions.vertical_flip(image)

            if vertical_flipped_image is not None:
                # Display the uploaded image
                container.image(vertical_flipped_image,
                                caption="Flipped Image", use_column_width=False)

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
                    contrast_image, caption="Contrast Image", use_column_width=False)

        brightness_input = st.slider(
            "Change Brightness:", min_value=-255, max_value=255, value=0, step=1, key='brightness_slider')

        # Check if the slider value has changed
        if st.button("Apply Brightness"):

            bright_image = functions.image_brightness(image, brightness_input)

            if bright_image is not None:
                # Display the uploaded image
                container.image(
                    bright_image, caption="Bright Image", use_column_width=False)

        # # Color space transform
        # st.text("Color space transformation")

        # options = ['RGB', 'HSV', 'LAB']

        # # Create a dropdown using selectbox
        # selected_option = st.selectbox('Transform color space:', options)

        # color_space_image = functions.color_space_transform(
        #         image, selected_option)

        # if color_space_image is not None:
        #     if selected_option is not None:
        #             container.image(
        #                 color_space_image, caption="Transformed Image", use_column_width=False)

        # Grayscale conversion

        st.text("Grayscale conversion:")
        if st.button("Convert to grayscale"):

            gray_image = functions.convert_to_grayscale(image)

            if gray_image is not None:
                # Display the uploaded image
                container.image(
                    gray_image, caption="Gray Image", use_column_width=False)

        # Thresholding

        st.text("Otsu thresholding:")
        if st.button("Apply threshold"):

            threshold_image = otsu.apply_otsu_thresholding(image)

            if threshold_image is not None:
                # Display the uploaded image
                container.image(
                    threshold_image, caption="Thresholded Image", use_column_width=False)

        # for quantisation
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

        # for histogram equalisation
        st.text("Histogram Equalisation:")
        if st.button("Apply Histogram Equalisation"):

            equalised_image = histogramEqualisation.histogram_equalization(
                image)

            if equalised_image is not None:
                # Display the uploaded image
                container.image(
                    equalised_image, caption="Equalised Image", use_column_width=False)

        # noise removal blurs
        st.text("Noise Removal Blurs:")

        with st.form("blurs_form"):

            kernel_size = st.number_input(
                "Enter the desired size of kernel:", value=0, step=1)
            st.text("Can only be an ODD integer")

            if (kernel_size % 2 == 0):
                st.warning("Please enter an odd value for kernel size")

            gauss = st.form_submit_button("Gaussian Blur")
            median = st.form_submit_button("Median Blur")
            # bilateral = st.form_submit_button("Bilateral Blur")

        if (gauss):

            gaussian_image = functions.gaussian_blur(image, kernel_size)

            if gaussian_image is not None:
                # Display the uploaded image
                container.image(
                    gaussian_image, caption="Blurred Image", use_column_width=False)

        if (median):

            median_image = functions.median_blur(image, kernel_size)

            if median_image is not None:
                # Display the uploaded image
                container.image(
                    median_image, caption="Blurred Image", use_column_width=False)

        # intensity transformations
        st.text("Apply Intensity Transformations:")

        st.text("Power law:")
        # Add a slider
        gamma = st.number_input(
            "Enter the Gamma value:", min_value=0.0, max_value=100.0, value=1.0, step=0.01)

        # Check if the slider value has changed
        if st.button("Apply Power Law"):

            power_image = intensity_transform_laws.power_law_transform(
                image, gamma)

            if power_image is not None:
                # Display the uploaded image
                container.image(
                    power_image, caption="Power Image", use_column_width=False)

        st.text("Log law:")
        c = st.number_input("Enter the C value:", min_value=0.0,
                            max_value=25.0, value=1.0, step=0.01)

        # Check if the slider value has changed
        if st.button("Apply Log Law"):

            log_image = intensity_transform_laws.log_transform(image, c)

            if log_image is not None:
                # Display the uploaded image
                container.image(
                    log_image, caption="log Image", use_column_width=False)

        st.text("Negative of image:")
        if st.button("Take negative of the image"):

            neg_image = intensity_transform_laws.negative_of_image(image)

            if neg_image is not None:
                # Display the uploaded image
                container.image(
                    neg_image, caption="Negative of Image", use_column_width=False)

        st.text("Watermarking:")
        # Create a form for watermarking
        with st.form("watermark_form"):
            # Allow the user to choose a watermark image
            watermark_image = st.file_uploader("Choose a watermark...", type=[
                                               "jpg", "jpeg", "png", "tiff"])

            # Allow the user to set transparency
            transparency_input = st.slider(
                "Choose transparency:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            # Add a submit button to the form
            submitted = st.form_submit_button("Add watermark")

        # Check if the form is submitted
        if submitted and watermark_image is not None:
            # Process the watermark and display the result
            wt_image = Image.open(watermark_image)
            watermarkedImg = waterMark.add_watermark(
                image, wt_image, transparency=transparency_input)
            st.text("HIIIIIIIIII")

            if watermarkedImg is not None:
                # Display the watermarked image
                st.text("HIIIIIIIIII")

                container.image(
                    watermarkedImg, caption="Watermarked Image", use_column_width=False)
                st.text("HIIIIIIIIII")


if __name__ == "__main__":
    main()
