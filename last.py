import csv
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import streamlit as st
import os
from PIL import Image
import queue
import threading

# Create a folder to store training images if it doesn't exist
os.makedirs("StudentDetails", exist_ok=True)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Counter for the number of images captured
image_counter = 0

# Flag to indicate when to stop capturing images
stop_capture = False

enrollment_num = ""
name = ""
dir_name = ""

# Initialize a queue for thread-safe communication
detected_ids_queue = queue.Queue()

# Set to store detected IDs
detected_ids = set()

# Initialize a lock for thread-safe access to detected_ids set
detected_ids_lock = threading.Lock()

if "capture" not in st.session_state:
    st.session_state.capture = False

if "mark" not in st.session_state:
    st.session_state.mark = False

def assignImagesAndLabels(path):
    # List all directories in the given path
    subdirectories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Compile a list of image paths from subdirectories
    imagePaths = [
        os.path.join(subdir, f)
        for subdir in subdirectories
        for f in os.listdir(subdir)
        if os.path.isfile(os.path.join(subdir, f))
    ]

    faces = []
    ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert("L")  # Convert image to grayscale
        imageNp = np.array(pilImage, "uint8")  # Convert PIL image to numpy array
        # Extract ID from the file name
        id = int(os.path.split(imagePath)[-1].split("_")[1])
        faces.append(imageNp)
        ids.append(id)

    return faces, ids

def trainModel(trainimage_path, trainimagelabel_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = assignImagesAndLabels(trainimage_path)
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainimagelabel_path)
    st.success("Images trained successfully!!")

def detect(frame):
    global detected_ids_queue

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")

    frm = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf < 50:  # Adjust confidence threshold as needed
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frm, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"ID: {id} with confidence: {conf}")
            detected_ids_queue.put(id)  # Add detected ID to the queue

        else:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frm, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print("Unknown face detected")

    return av.VideoFrame.from_ndarray(frm, format='bgr24')

def mark_attendance():
    global detected_ids

    def update_detected_ids():
        global detected_ids_lock  # Use the global lock
        while True:
            try:
                id = detected_ids_queue.get(timeout=1)  # Get detected ID from the queue
                with detected_ids_lock:
                    detected_ids.add(id)  # Add ID to the set within the lock
            except queue.Empty:
                continue

    # Start a thread to continuously update detected_ids from the queue
    thread = threading.Thread(target=update_detected_ids)
    thread.start()

    st.write(st.session_state.mark)
    web_ctx = webrtc_streamer(
        key="mark_attendance",
        video_frame_callback=detect,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
    )

    if web_ctx.state.playing == False:
        with detected_ids_lock:
            st.write(detected_ids)

def received(frame):
    global image_counter, stop_capture ,enrollment_num,name,dir_name

    frm = frame.to_ndarray(format="bgr24")

    if not stop_capture:
        faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Save the detected face as an image
            face_img = frm[y:y+h, x:x+w]
            sub_director = f"TrainingImage/{enrollment_num}_{name}"
            image_path = f"{sub_director}/{name}_{enrollment_num}_{image_counter}.jpg"
            cv2.imwrite(image_path, face_img)

            # Increment image counter
            image_counter += 1

            # Check if 50 images have been saved
            if image_counter >= 50:
                stop_capture = True
                break  # Exit the loop to stop further processing

    return av.VideoFrame.from_ndarray(frm, format='bgr24')

def main():
    global enrollment_num,name,dir_name
    st.title("Student Attendance System")

    choice = st.sidebar.selectbox("Choose Functionality", ["Take Image", "Train Image", "Track Attendance"])

    if choice == "Take Image":
        st.header("Take Image")
        enrollment_num = st.text_input("Enrollment Number")
        name = st.text_input("Name")
        if st.button("Capture Images"):
            st.session_state.capture = True
            try:
                dir_name = f"TrainingImage/{enrollment_num}_{name}"
                os.makedirs(dir_name)
                row = [enrollment_num, name]
                with open("StudentDetails/studentdetails.csv", "a+") as csvFile:
                    writer = csv.writer(csvFile, delimiter=",")
                    writer.writerow(row)
                    csvFile.close()
                st.toast("Images Saved for ER No:" + enrollment_num + " Name:" + name)
            except FileExistsError as F:
                st.error("Student already exists")

        if st.session_state.capture:
            st.write(f"Capture started for {name} with enrollment number {enrollment_num}")
            # Render the WebRTC component with the selected camera and microphone
            webrtc_streamer(
                key="example",
                video_frame_callback=received,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
            )

    elif choice == "Train Image":
            st.header("Train Image")
            if st.button("Train Images"):
                trainModel("TrainingImage","TrainingImageLabel/Trainner.yml")


    elif choice == "Track Attendance":
        st.header("Track Attendance")
        if st.button("Attend"):
            st.session_state.mark = True

        if st.session_state.mark:
            mark_attendance()

if __name__ == "__main__":
    main()
