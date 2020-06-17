# imports
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import time
import functools
import warnings
# main code
def main():
    st.title("Neural Style Transfer")
    st.sidebar.title("Menu bar : Neural Style Transfer")
    st.markdown("Wanna add some style 🖼️ to your clicks 📷 in funky programming way..")
    st.sidebar.markdown("Apply filters to your clicks 📷.")
    st.sidebar.subheader(" Please do read How to guide.")

    if st.sidebar.checkbox('Show How to use and README.'):
        st.header("HOW TO . . .")
        st.subheader("If you won't paste the URL and File name the application will raise the TypeError")
        st.subheader("Enter the information in given input fields. \nTo upload your content URL use 'https://postimages.org/'")
        st.markdown("After going on the site given above upload your image on which you want to apply style tranfer")
        st.image("info1.jpg")
        st.markdown("Copy the 'Direct Link' and paste it into the 'URL input section' and paste the file name in the link which is at the end of 'Direct link' into the File name section and hit enter.")
        st.markdown("After doing this procedure the application will take some time to process the image and apply the style.\n You can select diferent filters giiven in side menu bar.")
        st.markdown("Once the style is applied to you image you can manually save your ready image by just right clicking on image and select 'save as...' ")

    if st.sidebar.checkbox("Enter Filename and URL"):
        file_name = st.text_input("File name goes here")
        url = st.text_input("URL of input content image goes here")


    if file_name != "" and url != "":
        content_path = tf.keras.utils.get_file(file_name, url)
        #content_path = tf.keras.utils.get_file('IMG-20161026-161718.jpg', 'https://i.postimg.cc/NGXBFxkw/IMG-20161026-161718.jpg')
        style_path = tf.keras.utils.get_file('The_Verge_GOT_Portrait_Wallpaper.0.png','https://cdn.vox-cdn.com/uploads/chorus_asset/file/16282418/The_Verge_GOT_Portrait_Wallpaper.0.png')

        def tensor_to_image(tensor):
            tensor = tensor*255
            tensor = np.array(tensor, dtype=np.uint8)
            if np.ndim(tensor)>3:
                assert tensor.shape[0] == 1
                tensor = tensor[0]
            return PIL.Image.fromarray(tensor)

        @st.cache(persist=True)
        def load_img(path_to_img):
            max_dim = 512
            img = tf.io.read_file(path_to_img)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)

            shape = tf.cast(tf.shape(img)[:-1], tf.float32)
            long_dim = max(shape)
            scale = max_dim / long_dim

            new_shape = tf.cast(shape * scale, tf.int32)

            img = tf.image.resize(img, new_shape)
            img = img[tf.newaxis, :]
            return img

        @st.cache(persist=True)
        def imshow(image, title=None):
            if len(image.shape) > 3:
                image = tf.squeeze(image, axis=0)

            #plt.imshow(image)
            #if title:
            #    plt.title(title)

        content_image = load_img(content_path)
        style_image = load_img(style_path)

        #plt.subplot(1, 2, 1)
        #imshow(content_image, 'Content Image')
        #plt.subplot(1, 2, 2)
        #imshow(style_image, 'Style Image')

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        if st.sidebar.checkbox("Show Stylized Image"):
            st.image(tensor_to_image(stylized_image))









if __name__ == '__main__':
    main()
