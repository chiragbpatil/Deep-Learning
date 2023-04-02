import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import os
import recognize_digits

stroke_width = 5
drawing_mode = 'freedraw'
stroke_color = '#FFFFFF'
bg_color = '#000000'
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    # background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,
    width=500,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)


if st.button('Predict'):
   img = canvas_result.image_data#.astype('uint8')
   img_path = os.path.join("tmp","temp.jpg")
   cv2.imwrite(img_path,img)
   prediction,probs = recognize_digits.recognize(img_path)
   st.title(f'Result: {prediction}',)
   
   chart_data = pd.DataFrame(probs,columns=[0,1,2,3,4,5,6,7,8,9])
   st.bar_chart(chart_data,)
   