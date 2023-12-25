import base64
import streamlit as st
from PIL import Image

import utils.functions as functions
import utils.intensity_transforms as intensity_transforms
import utils.histogram_equalisation as histogram_equalisation
import utils.quantisation as quantisation
import utils.otsu as otsu
import utils.watermark as watermark

# Declare image variable outside the main function
image = None


def custom_navbar(logo_path):
    st.markdown(f"""
    <nav style="background-color: #333; padding: 10px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <div style="transition: background-color 0.3s;" onmouseover="this.style.backgroundColor='#444'" onmouseout="this.style.backgroundColor='#333'">
            <img src="data:image/png;base64,{logo_path}" alt="Logo" style="height:75px; margin-left: 40px; border-radius: 10px;">
        </div>
        <h2 style='color: white; margin: 0; text-shadow: 2px 2px 4px #000000;'>Phomo App Project</h2>
    </nav>
    """, unsafe_allow_html=True)



def main():
    
    if image not in st.session_state:
        st.session_state.image = None
    
    logo_path = "public/images/logo.png"
    with open(logo_path, "rb") as img_file:
        logo = base64.b64encode(img_file.read()).decode()

    custom_navbar(logo)

    with st.sidebar:
        uploaded_image = st.file_uploader("Upload Image",label_visibility="collapsed", type=["jpg", "jpeg", "png", "tiff"])

    if uploaded_image is not None:
        st.session_state.image = Image.open(uploaded_image)
        
    if st.session_state.image is not None:
        width, height = st.session_state.image.size
        container = st.image(st.session_state.image, caption="Uploaded Image", use_column_width="auto")
        
        # Get image bytes and format
        with st.sidebar:
            img_bytes, img_format = functions.get_image_bytes(st.session_state.image)
            mime_type = f"image/{img_format}"

        # Add a download button
            st.download_button(
            label="Download Image",
            data=img_bytes,
            file_name=f"processed_image.{img_format}",
            mime=mime_type,
            use_container_width=True
        )


    with st.sidebar:
        st.header("Functions:")
        with st.expander("Resize"):
            with st.form("resize_form", border=False):
                width_input = st.number_input("Enter the desired width:", value=0, step=1)
                height_input = st.number_input("Enter the desired height:", value=0, step=1)
                submitted = st.form_submit_button("Resize", use_container_width=True)
                if submitted and width_input != 0 and height_input != 0 and st.session_state.image:
                    st.session_state.image = functions.resize_image(st.session_state.image, width_input, height_input)
                    container.image(st.session_state.image, caption="Resized Image", use_column_width="auto")

                elif submitted:
                    st.warning("Please enter both width and height before resizing.")
    
        with st.expander("Rotate"):
            with st.form("rotate_form", border=False):
                angle = st.number_input("Enter the desired angle:", value=0, step=1)
                uncropped_rotate = st.form_submit_button("Rotate without crop", use_container_width=True)
                cropped_rotate = st.form_submit_button("Rotate and Crop", use_container_width=True)
                if uncropped_rotate and st.session_state.image is not None:
                    st.session_state.image = functions.rotate_not_cropped(st.session_state.image, angle)
                    container.image(st.session_state.image, caption="Cropped Image", use_column_width="auto")

                if cropped_rotate and st.session_state.image is not None:
                    st.session_state.image = functions.rotate_cropped(st.session_state.image, angle)
                    container.image(st.session_state.image, caption="Cropped Image", use_column_width="auto")

        # Flip the image
        with st.expander("Flips"):
            horizontal_flip = st.button("Horizontal Flip", use_container_width=True)
            if horizontal_flip:
                st.session_state.image = functions.horizontal_flip(st.session_state.image)
                container.image(st.session_state.image, caption="Flipped Image", use_column_width="auto")

            vertical_flip = st.button("Vertical Flip",  use_container_width=True)
            if vertical_flip:
                st.session_state.image = functions.vertical_flip(st.session_state.image)
                container.image(st.session_state.image, caption="Flipped Image", use_column_width="auto")


        # Contrast and Brightness Adjustments
        with st.expander("Brightness and Contrast"):
            contrast_input = st.slider("Change Contrast:", min_value=0, max_value=255, value=125, step=1, key='contrast_slider')
            if st.button("Apply Contrast", use_container_width=True):
                st.session_state.image = functions.linearContrastStretch(st.session_state.image, contrast_input)
                container.image(st.session_state.image, caption="Contrast Image", use_column_width="auto")

            brightness_input = st.slider("Change Brightness:", min_value=-255, max_value=255, value=0, step=1, key='brightness_slider')
            if st.button("Apply Brightness", use_container_width=True):
                st.session_state.image = functions.image_brightness(st.session_state.image, brightness_input)
                container.image(st.session_state.image, caption="Bright Image", use_column_width="auto")


        # Grayscale Conversion
        if st.button("Greyscale", use_container_width=True):
            st.session_state.image = functions.convert_to_grayscale(st.session_state.image)
            container.image(st.session_state.image, caption="Gray Image", use_column_width=True)


        # Otsu Thresholding
        if st.button("Otsu Threshold", use_container_width=True):
            st.session_state.image = otsu.apply_otsu_thresholding(st.session_state.image)
            container.image(st.session_state.image, caption="Thresholded Image", use_column_width="auto")

        with st.expander("Quantisation"):
            # Quantisation
            quantisation_input = st.slider("Quantise the image:", min_value=2, max_value=256, value=16, step=1, key='quantisation_slider')
            if st.button("Apply Quantisation"):
                st.session_state.image = quantisation.quantize_image(st.session_state.image, quantisation_input)
                container.image(st.session_state.image, caption="Quantised Image", use_column_width="auto")


        # Histogram Equalisation
        if st.button("Apply Histogram Equalisation", use_container_width=True):
            st.session_state.image = histogram_equalisation.histogram_equalization(st.session_state.image)
            container.image(st.session_state.image, caption="Equalised Image", use_column_width="auto")

        # Noise Removal Blurs
        with st.expander("Blurs"):
            with st.form("blurs_form", border=False):
                kernel_size = st.number_input("Enter the desired size of kernel:", value=0, step=1)
                st.text("Can only be an ODD integer")
                if kernel_size % 2 == 0:
                    st.warning("Please enter an odd value for kernel size")

                gauss = st.form_submit_button("Gaussian Blur", use_container_width=True)
                median = st.form_submit_button("Median Blur", use_container_width=True)

            if gauss:
                st.session_state.image = functions.gaussian_blur(st.session_state.image, kernel_size)
                container.image(st.session_state.image, caption="Blurred Image", use_column_width="auto")

            if median:
                st.session_state.image = functions.median_blur(st.session_state.image, kernel_size)
                container.image(st.session_state.image, caption="Blurred Image", use_column_width="auto")

        # Intensity Transformations
        with st.expander("Intensity Transforms"):
            st.text("Power law:")
            gamma = st.number_input("Enter the Gamma value:", min_value=0.0, max_value=100.0, value=1.0, step=0.01)
            if st.button("Apply Power Law", use_container_width=True):
                st.session_state.image = intensity_transforms.power_law_transform(st.session_state.image, gamma)
                container.image(st.session_state.image, caption="Power Image", use_column_width="auto")

            st.text("Log law:")
            c = st.number_input("Enter the C value:", min_value=0.0, max_value=25.0, value=0.0, step=0.01)
            if st.button("Apply Log Law", use_container_width=True):
                st.session_state.image = intensity_transforms.log_transform(st.session_state.image, c)
                container.image(st.session_state.image, caption="log Image", use_column_width="auto")

            st.text("Negative of image:")
            if st.button("Take negative of the image", use_container_width=True):
                st.session_state.image = intensity_transforms.negative_of_image(st.session_state.image)
                container.image(st.session_state.image, caption="Negative of Image", use_column_width="auto")

        # Watermarking
        with st.expander("Watermark"):
            with st.form("watermark_form", border=False):
                watermark_image = st.file_uploader("Select Watermark Image", type=["jpg", "jpeg", "png", "tiff"])
                submitted = st.form_submit_button("Apply watermark")

            if submitted and watermark_image is not None:
                wt_image = Image.open(watermark_image)
                st.session_state.image = watermark.add_watermark(st.session_state.image, wt_image)
                container.image(st.session_state.image, caption="Watermarked Image", use_column_width="auto")
                
        

if __name__ == "__main__":
    main()
