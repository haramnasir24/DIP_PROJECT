import dearpygui.dearpygui as dpg

dpg.create_context()


def file_dialog_callback(sender, app_data, user_data):
    selected_file_path = app_data["file_path_name"]
    if selected_file_path:
        show_image_window(selected_file_path)

# Load the image and get its dimensions


def show_image_window(image_path):

    width, height, channels, data = dpg.load_image(image_path)
    with dpg.window(label="Image Display", width=width, height=height, id="ImageDisplay", no_move=True, no_resize=True):
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(
                width=width, height=height, default_value=data, tag="texture_tag")

        # Add an image widget to display the selected image
        dpg.add_image("texture_tag")
        
        # Adjust the position of the Image Display window
        dpg.set_item_pos("ImageDisplay", [400, 100])  # Adjust the coordinates as needed


# to select a file
with dpg.file_dialog(directory_selector=False, show=False, callback=file_dialog_callback, id="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".png")
    dpg.add_file_extension(".jpg")
    dpg.add_file_extension(".jpeg")

with dpg.window(label="Menu", width=250, height=800, id="Menu", no_move=True, no_resize=True):
    dpg.add_button(label="Open Image",
                callback=lambda: dpg.show_item("file_dialog_id"))

    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save")
            dpg.add_menu_item(label="Save As")

        with dpg.menu(label="Settings"):
            dpg.add_menu_item(label="Setting 1")
            dpg.add_menu_item(label="Setting 2")

        dpg.add_menu_item(label="Help")

        with dpg.menu(label="Widgets"):
            dpg.add_checkbox(label="Pick Me")
            dpg.add_button(label="Press Me")
            dpg.add_color_picker(label="Color Me")

dpg.create_viewport(title='Image Editor', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
