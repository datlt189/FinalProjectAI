import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import time
import matplotlib.pyplot as plt
from os import listdir
from keras.utils import load_img
from keras.utils.image_utils import img_to_array

# Load model
model = load_model('vehicle_counting_and_classification.h5')

def main():
    # Title the web
    st.markdown('<h1 style="color: BlueViolet;">VEHICLE COUNTING AND CLASSIFICATION</h1>', unsafe_allow_html=True)

    # Create menu tab
    tabs = ["HOME", "INFORMATION"]
    active_tab = st.sidebar.radio("Select tab", tabs)

    # Display the content (depend on the following tab)
    if active_tab == "HOME":
        home_tab()
    elif active_tab == "INFORMATION":
        information_tab()

# Home
def home_tab():
    # Display the title, header, note and "Drag and drop" bar
    st.header("Detect, count and classify the vehicles in the picture")
    st.markdown('<p style="color: orange;">\u26A0️ Please pay attention to the following notes about the image to be processed:</p>', unsafe_allow_html=True)
    st.write("\u2611 _The size SHOULD be of moderate dimensions (around 1280x720)._")
    st.write("\u2611 _MUST have clean background._")
    st.write("\u2611 _The objects (vehicles, things, etc) MUST stand apart at a distance._")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png"])
    
    # Process the file (if true)
    if uploaded_file is not None:
        st.success("Upload image successfully!")
        # Create button
        click = st.button("Click here to process right now!")
        img_PIL = Image.open(uploaded_file)
        img_array = np.array(img_PIL) # np.array -> RGB
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        #Process if click
        if(click):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Blur image
            edges = cv2.Canny(blurred, 30, 130) # Edge detection in image

            # Dilate to increase the dimension
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Find contour
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #EXTERNAL, TREE,...

            # Divide image into subimages based on detected contours
            subimages = []
            for contour in contours:
                if cv2.contourArea(contour) > 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    subimg = img[y:y+h, x:x+w]
                    subimages.append((subimg, (x, y, w, h)))

            # Classify subimages using pre-trained CNN model and draw bounding box
            vehicle_count = 0
            for subimg, (x, y, w, h) in subimages:
                resized = cv2.resize(subimg, (100, 100)) # Size of img in model
                resized = np.expand_dims(resized, axis=0)
                resized = resized.astype('float32') / 255
                pred = (model.predict(resized).argmax())
                class_name = ['', 'Motorbike', 'Car']
                if class_name[pred] == 'Motorbike':
                    vehicle_type = 'Motorbike'
                    color = (0, 255, 0) 
                elif class_name[pred] == 'Car':
                    vehicle_type = 'Car'
                    color = (255, 0, 0) 
                else:
                    pass

                # Vẽ hình chữ nhật giới hạn
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

                # Hiển thị nhãn loại phương tiện
                label = f'{vehicle_type}'
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


                vehicle_count += 1

            # Display the number of detected vehicles
            text = f"Detected vehicles: {vehicle_count}"
            cv2.putText(img, text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            st.image(img, channels="BGR", caption="Result")
# Tab
def information_tab():
    st.header(":point_down: Check my Git for more")
    st.markdown("[Link is here](https://github.com/datlt189/FinalProjectAI)")

if __name__ == "__main__":
    main()
