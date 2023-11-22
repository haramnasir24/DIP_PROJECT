import dearpygui.dearpygui as dpg

dpg.create_context()


def file_dialog_callback(sender, app_data, user_data):
    selected_file_path = app_data["file_path_name"]
    if selected_file_path:
        show_image_window(selected_file_path)

# Load the image and get its dimensions


def show_image_window(image_path):

    width, height, channels, data = dpg.load_image(image_path)
    with dpg.window(label="Image Display", width=width, height=height):
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(
                width=width, height=height, default_value=data, tag="texture_tag")

        # Add an image widget to display the selected image
        dpg.add_image("texture_tag")


# to select a file
with dpg.file_dialog(directory_selector=False, show=False, callback=file_dialog_callback, id="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".png")
    dpg.add_file_extension(".jpg")
    dpg.add_file_extension(".jpeg")

with dpg.window(label="Menu", width=300, height=500):
    dpg.add_button(label="Open Image",
                   callback=lambda: dpg.show_item("file_dialog_id"))

dpg.create_viewport(title='Image Editor', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
