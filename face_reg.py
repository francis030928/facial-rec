import cv2
import streamlit as st

st.markdown("<h1 style = 'text-align: right; color: #B2A4FF'>FACIAL DETECTION</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: right; color: #FAF8F1'>Built by GoMyCode Pumpkin Reedemers</h6>", unsafe_allow_html = True)


img1 = st.image('pngwing.com (1).png', caption = "FACE OF REDEEMERS", width = 500)

st.write('Pls register your name')
username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}.")

# if st.button('camera'):


# Add instructions to the Streamlit app interface
st.write("Welcome to Face Detection using Viola-Jones Algorithm!")
st.write("Instructions:")
st.write("1. Position yourself in front of the camera.")
st.write("2. The app will detect your face and draw rectangles around it.")
st.write("3. Adjust the parameters below to customize the detection.")
st.write("4. Press 'q' to exit the app.")

# Add a feature to save the images with detected faces
if st.button('Save Images'):
    st.write("Saving images...")

# Add a feature to adjust the minNeighbors parameter
min_neighbors = st.slider("Adjust minNeighbors", 1, 10, 5)

# Add a feature to adjust the scaleFactor parameter
scale_factor = st.slider("Adjust scaleFactor", 1.1, 2.0, 1.3)

if st.button('camera'): 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()   #....................................... Initiate the camera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #.................. Grayscale it using the cv grayscale library

    #   Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor= scale_factor, minNeighbors= min_neighbors, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    #   Draw rectangles around the detected faces
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (225, 255, 0), 2)

    # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
    
    # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    # Release the webcam and close all windows
    camera.release()
    cv2.destroyAllWindows()

# # Objective
# # Improving the Streamlit app for face detection using Viola-Jones algorithm of the example of the content

# # Instruction
# # Add instructions to the Streamlit app interface to guide the user on how to use the app.
# # Add a feature to save the images with detected faces on the user's device.
# # Add a feature to allow the user to choose the color of the rectangles drawn around the detected faces.
# # Add a feature to adjust the minNeighbors parameter in the face_cascade.detectMultiScale() function.
# # Add a feature to adjust the scaleFactor parameter in the face_cascade.detectMultiScale() function.


# # Hints:
# # Use the st.write() or st.markdown() functions to add instructions to the interface.

# # Use the cv2.imwrite() function to save the images.
# # Use the st.color_picker() function to allow the user to choose the color of the rectangles.
# # Use the st.slider() function to allow the user to adjust the minNeighbors parameter.
# # Use the st.slider() function to allow the user to adjust the scaleFactor parameter.

# import cv2
# import streamlit as st
# import numpy as np

# # # Function to perform face detection using Viola-Jones algorithm
# # def detect_faces(image, minNeighbors, scaleFactor, color):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
# #     for (x, y, w, h) in faces:
# #         cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
# #     return image

# # Streamlit app
# def main():
#     st.title("Face Detection App")
#     st.write("Upload an image to detect faces")

#     # Upload image
#     uploaded_file = st.file_uploader("Choose File", type=['jpg', 'jpeg', 'png', 'gif'])
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)

#         # Display original image
#         st.image(image, channels="BGR", caption="Original Image")

#         # Instructions for face detection
#         st.markdown("### Step 1: Detect Faces")
#         st.write("Click the button below to detect faces in the uploaded image")
#         if st.button("Detect Faces"):
#             # Get user-adjusted parameters
#             minNeighbors = st.slider("minNeighbors", min_value=1, max_value=10, step=1, value=5)
#             scaleFactor = st.slider("scaleFactor", min_value=1.1, max_value=4.0, step=0.1, value=1.2)
#             color = st.color_picker("Rectangle Color", value="#FF0000")

#             # Perform face detection
#             image_with_faces = detect_faces(image, minNeighbors, scaleFactor, color)

#             # Display image with detected faces
#             st.image(image_with_faces, channels="BGR", caption="Image with Detected Faces")

#             # Save image with detected faces
#             if st.button("Save Image"):
#                 cv2.imwrite("image_with_faces.jpg", image_with_faces)
#                 st.write("Image with detected faces saved successfully!")

# if __name__ == '__main__':
#     main()


# import cv2
# # import numpy as np
# import streamlit as st

# # Load the pre-trained face cascade classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Initialize the webcam
# camera = cv2.VideoCapture(0)

# # Add instructions to the Streamlit app interface
# st.write("Welcome to Face Detection using Viola-Jones Algorithm!")
# st.write("Instructions:")
# st.write("1. Position yourself in front of the camera.")
# st.write("2. The app will detect your face and draw rectangles around it.")
# st.write("3. Adjust the parameters below to customize the detection.")
# st.write("4. Press 'q' to exit the app.")

# # Add a feature to save the images with detected faces
# if st.button('Save Images'):
#     st.write("Saving images...")
# count = 1
# while True:
#     frame = camera.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
#     for (x, y, width, height) in faces:
#         cv2.rectangle(frame, (x, y), (x + width, y + height), (225, 255, 0), 2)
#     cv2.imwrite(f'face{count}.png', frame[y:y + height, x:x + width])
#     count += 1
#     cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Add a feature to allow the user to choose the color of the rectangles
# color = st.color_picker("Choose rectangle color", "#00ff00")

# # Add a feature to adjust the minNeighbors parameter
# min_neighbors = st.slider("Adjust minNeighbors", 1, 10, 5)

# # Add a feature to adjust the scaleFactor parameter
# scale_factor = st.slider("Adjust scaleFactor", 1.1, 2.0, 1.3)

# while True:
#     frame = camera.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
#     for (x, y, width, height) in faces:
#         cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
#     cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Release the webcam and close all windows
# camera.release()
# cv2.destroyAllWindows()
