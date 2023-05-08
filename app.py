
import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow import keras
from keras import models
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing import image
import cv2
import base64
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def load_image(image_file):
    img = Image.open(image_file)
    return img

def pred_result(img):
    lesion_dict = {
        'gt':'Glioma Tumour',
        'nt':'No tumour',
        'mt':'Meningioma tumour',
        'pt':'Pituitary tumour'
        }
    lesion_class_dict = {
        0 : 'gt',
        1 : 'nt',
        2 : 'mt',
        3 : 'pt'
        }
    cancer_class = {
        0 : 'Cancerous',
        1 : 'Non-Cancerous',
        2 : 'Cancerous',
        3 : 'Cancerous'
        }

    model = tf.keras.models.load_model("C:\\Users\\chinm\\OneDrive\\Desktop\\College Projects and Saved Models\\Major-1 Sem7\\Model\\EfficientNetB3.h5")

    img = np.array(img)
    img = cv2.resize(img,(160,160))
    img = img.reshape(1,160,160,3)
    p = model.predict(img)
    p = np.argmax(p,axis=1)[0]

    if p==0:
        p='Glioma Tumor'
    elif p==1:
        p='No tumor'
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'

    return p
    

def capture():
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0


    #run = st.checkbox('Run')
    while True:
        _, frame = camera.read()
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256 == 32:
            #space pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            
        elif k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    #st.write(type(frame))
    image_view = cv2.imread(r"C:\\Users\\chinm\\opencv_frame_0.png")
    image_view_arr = np.asarray(image_view)
    #st.write(type(image_view_arr))
    frame = cv2.cvtColor(image_view_arr, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)



def main():

    st.title("Brain Tumour Prediction")
    menu = ["How to use", "Image Capture", "Image Upload", "About"]
    choice = st.sidebar.selectbox("Menu",menu)
 #   if choice == "EDA":
  #       st.subheader("EDA")
   #  elif choice == "Image Upload":
    #     st.subheader("Image Upload")
     # elif choice == "Image Capture":
      #    st.subheader("Image Capture")


    if choice == "How to use":
        st.subheader("How to use")
        st.write("To capture the image : Press space")
        st.write("To exit the webcam window : Press esc")
        

    #elif choice == "Exploratory Data Analysis":
    #    st.subheader("Exploratory Data Analysis")
    #    df = pd.read_csv("C:\\Users\\risha\\OneDrive\\Desktop\\HAM.csv", nrows = 1000)
    #    #st.write(df)
    #    st.bar_chart(df['localization'])
    #    df = pd.read_csv("C:\\Users\\risha\\OneDrive\\Desktop\\HAM.csv")
    #    x = df['age']
    #    y = df['cell_type_idx']
    #    fig=plt.figure(figsize = (4,3))
    #    plt.scatter(x,y)
    #    plt.xlabel("AGE")
    #    plt.ylabel("Cell_type")
    #    #st.baloons()
    #    st.pyplot(fig)

        
    elif choice == "Image Upload":
        st.subheader("Image Upload")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                            "filesize":image_file.size}
            st.write(file_details)
            # To View Uploaded Image
            img = load_image(image_file)
            st.image(img,width=250)
            img.save(r'C:\Users\risha\img_updated.png')
            img = image.load_img(r'C:\Users\risha\img_updated.png', target_size = (160,160,3))
            
            p = pred_result(img)
            st.write('You are suffering from : '+ p)
            
            
    elif choice == "Image Capture":
        st.subheader("Webcam Live Feed")
        
        capture()
        
        image_path = r"C:\\Users\\risha\\opencv_frame_1.png"
        img = image.load_img(image_path, target_size = (160,160,3))

        p = pred_result(img)
        st.write('You are suffering from : '+ p)
                    
    elif choice == "About":
        st.subheader("About")
        st.text("A brain tumor is a collection, or mass, of abnormal cells in your brain. \nYour skull, which encloses your brain, is very rigid. Any growth \ninside such a restricted space can cause problems. Brain \ntumors can be cancerous (malignant) or noncancerous (benign). When \nbenign or malignant tumors grow, they can cause the pressure inside your skull \nto increase. This can cause brain damage, and it can be life-threatening. \nThe main three types of Brain Tumour are:-")        
        st.text("1. Meningioma Tumour \n2. Glioma Tumour \n3. Pituitary Tumour")
        st.text("Our research aims to detect the tumors in the early stages \nusing Magnetic Resonance Imaging (MRI) in order to better comprehend the \nstage of the malignancy. The MRI scan has not caused any harm to \nthe human body, and it is non-invasive, free of radiation damage, \nmulti-directional, and multi-dimensional identification are more accurate than \nthose done by CT and X-ray machines, for example. Medical professionals \nmostly employ the MRI image segmentation method.")

main()
