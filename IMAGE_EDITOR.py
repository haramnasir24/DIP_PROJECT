import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import functions
import intensity_transform_laws

def custom_title():
    st.markdown("""
        <h2 style='text-align: center; color: darkgreen; margin-top: 0px'>PHOTO EDITOR/PREPROCESSOR</h2>
    """, unsafe_allow_html=True)

global image
image = None
global container
container = None

def main():
    # st.title("PHOTO EDITOR/PREPROCESSOR")
    custom_title()



    with st.sidebar:

        # Allow the user to upload an image
        uploaded_image = st.file_uploader("**Choose an image...**", type=["jpg", "jpeg", "png", "tiff"])


    if uploaded_image is not None:


        

        # Display the uploaded image
        image = Image.open(uploaded_image)

        width, height = image.size

        if (width > 800 or height > 500):

            container = st.image(image, caption="Uploaded Image", use_column_width=True)
        
        else:
             
            container = st.image(image, caption="Uploaded Image", use_column_width=False)
    

    with st.sidebar:

        st.header("Functions:")

        st.markdown("**To Resize:**")
        with st.form("resize_form"):
            width_input = st.number_input("Enter the desired width:", min_value=1, value=1, step=1)
            height_input = st.number_input("Enter the desired height:", min_value=1, value=1, step=1)

            # Add a submit button to the form
            submitted = st.form_submit_button("Resize")

            # Check if the form is submitted and both values are provided
            if submitted and width_input != 0 and height_input != 0:
                st.write(f"Resizing to width: {width_input}, height: {height_input}")


                resized_image = functions.resize_image(image, width_input, height_input)

                image = resized_image

                if image is not None:
                    # Display the uploaded image
                    container.image(image, caption="Resized Image", use_column_width=False)

            elif submitted:
                st.warning("Please enter both width and height before resizing.")

        
        st.markdown("**To Rotate:**")

        with st.form("rotate_form"):

            angle = st.number_input("Enter the desired angle:", value=0, step=1)

            uncropped_rotate = st.form_submit_button("Rotate but NOT Crop")
            cropped_rotate = st.form_submit_button("Rotate AND Crop")


            if(uncropped_rotate):

                uncropped_rotated_image = functions.rotate_not_cropped(image, angle)

                image = uncropped_rotated_image

                if image is not None:
                    # Display the uploaded image
                    container.image(image, caption="Cropped Image", use_column_width=False)
            
            if(cropped_rotate):

                cropped_rotated_image = functions.rotate_cropped(image, angle)

                image = cropped_rotated_image

                if cropped_rotated_image is not None:
                    # Display the uploaded image
                    container.image(cropped_rotated_image, caption="Cropped Image", use_column_width=False)


        st.markdown("**Flip the image:**")
        horizontal_flip = st.button("Horizontal Flip")

        if(horizontal_flip):

                horizontal_flipped_image = functions.horizontal_flip(image)

                image = horizontal_flipped_image

                if horizontal_flipped_image is not None:
                    # Display the uploaded image
                    container.image(horizontal_flipped_image, caption="Flipped Image", use_column_width=False)


        vertical_flip = st.button("Vertical Flip")

        if(vertical_flip):

                vertical_flipped_image = functions.vertical_flip(image)

                image = vertical_flipped_image

                if vertical_flipped_image is not None:
                    # Display the uploaded image
                    container.image(vertical_flipped_image, caption="Flipped Image", use_column_width=False)
                
        
        st.markdown("**Grayscale Conversion:**")
        gray = st.button("Convert to grayscale")

        if(gray):

                gray_image = functions.rgb_to_grayscale(image)

                image = gray_image

                if gray_image is not None:
                    # Display the uploaded image
                    container.image(gray_image, caption="Flipped Image", use_column_width=False)

        
        st.markdown("**Contrast & Brightness:**")
        # Add a slider
        contrast_input = st.slider("Change Contrast:", min_value=0, max_value=255, value=125, step=1, key='contrast_slider')

        # Check if the slider value has changed
        if st.button("Apply Contrast"):

            
            contrast_image = functions.linearContrastStretch(image, contrast_input)

            image = contrast_image

            if contrast_image is not None:
                    # Display the uploaded image
                    container.image(contrast_image, caption="Contrast Image", use_column_width=False)



        brightness_input = st.slider("Change Brightness:", min_value=-255, max_value=255, value=0, step=1, key='brightness_slider')

        # Check if the slider value has changed
        if st.button("Apply Brightness"):
            
            
            bright_image = functions.image_brightness(image, brightness_input)

            image = bright_image

            if bright_image is not None:
                    # Display the uploaded image
                    container.image(bright_image, caption="Bright Image", use_column_width=False)


        st.markdown("**Noise Removal Blurs:**")

        with st.form("blurs_form"):

            kernel_size = st.number_input("Enter the desired size of kernel:",min_value=1, value=1, step=2)
            st.text("Can only be an ODD integer")


            guass = st.form_submit_button("Guassian Blur")
            median= st.form_submit_button("Median Blur")
            # bilateral = st.form_submit_button("Bilateral Blur")

        if(guass):

                guassian_image = functions.guassian_blur(image, kernel_size)

                image = guassian_image

                if guassian_image is not None:
                    # Display the uploaded image
                    container.image(guassian_image, caption="Blurred Image", use_column_width=False)


        if(median):

                median_image = functions.median_blur(image, kernel_size)

                image = median_image

                if median_image is not None:
                    # Display the uploaded image
                    container.image(median_image, caption="Blurred Image", use_column_width=False)


        # if(bilateral):

        #         bilate_image = functions.bilateral_blur(image)


        #         if bilate_image is not None:
        #             # Display the uploaded image
        #             container.image(bilate_image, caption="Rotated Image", use_column_width=False)
                    

        # st.markdown("**Apply Intensity Transformations:**")
        st.markdown("**Apply Power Law:**")
        # Add a slider
        gamma = st.number_input("Enter the Gamma value:", min_value=0.0, max_value=100.0, value=1.0, step=0.01)

        # Check if the slider value has changed
        if st.button("Apply Power Law"):

            
            power_image = intensity_transform_laws.power_law_transform(image, gamma)

            image = power_image

            if power_image is not None:
                    # Display the uploaded image
                    container.image(power_image, caption="Power Law Image", use_column_width=False)
        
        
        # c = st.number_input("Enter the C value:", min_value=0.0, max_value=25.0, value=1.0, step=0.01)

        # # Check if the slider value has changed
        # if st.button("Apply Log Law"):

            
        #     log_image = intensity_transform_laws.log_transform(image, c)

        #     image = log_image

        #     if log_image is not None:
        #             # Display the uploaded image
        #             container.image(log_image, caption="Log Law Image", use_column_width=False)
        
        st.markdown("**Convert to negative image:**")
        if st.button("Negative of the image"):

            
            neg_image = intensity_transform_laws.negative_of_image(image)

            image = neg_image

            if neg_image is not None:
                    # Display the uploaded image
                    container.image(neg_image, caption="Negative Image", use_column_width=False)


# Add a download button for the currently displayed image
    if container is not None:
        # Use st.download_button to trigger the download
        st.download_button(
            label="Download Image",
            data=image_to_bytes(image),
            file_name="downloaded_image.png",
            key="download_image",
        )

def image_to_bytes(img):
    img_data = BytesIO()
    img.save(img_data, format="PNG")
    img_bytes = img_data.getvalue()
    return img_bytes


if __name__ == "__main__":
    main()
