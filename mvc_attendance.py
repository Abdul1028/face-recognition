import csv
import pandas as pd
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import streamlit as st
import os
from PIL import Image
from typing import List, NamedTuple
import queue

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

detected_ids = set()



if "capture" not in st.session_state:
    st.session_state.capture = False

if "mark" not in st.session_state:
    st.session_state.mark = False

import threading

class Detection(NamedTuple):
    Id: int

lock = threading.Lock()
img_container = {"img": None}


result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
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
    print(subdirectories)
    print(imagePaths)

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

    global detected_ids

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")

    frm = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf < 80:  # Adjust confidence threshold as needed
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frm, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # st.session_state.detected_ids.add(id)
            detected_ids.add(id)
            result_queue.put(Detection(id))
            with lock:
                img_container["img"] = id

        else:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frm, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print("Unknown face detected")

    return av.VideoFrame.from_ndarray(frm, format='bgr24')

def mark_attendance():
    latest_set = set()
    st.write(st.session_state.mark)


    web_ctx = webrtc_streamer(
        key="mark_attendance",
        video_frame_callback=detect,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
    )


    if web_ctx.state.playing:
        labels_placeholder = st.empty()
        global detected_ids

        result = result_queue.get()

        df = pd.read_csv("StudentDetails/studentdetails.csv")
        col_names = ["Enrollment", "Name"]
        attendance = pd.DataFrame(columns=col_names)
        st.toast(result.Id)
        aa = df.loc[df["Enrollment"] == result.Id]["Name"].values

        from datetime import datetime

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        row = [aa,formatted_datetime]
        with open("Attendance/all_subs.csv", "a+") as csvFile:
            writer = csv.writer(csvFile, delimiter=",")
            writer.writerow(row)
            csvFile.close()
        st.toast(f"Attendance recored for {aa} you can leave")



        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
            a = int(result.Id)
            # b = set()
            # b.add(a)
            # detected_ids.add(a)
            # print("from b ",detected_ids)
            # with lock:
            #     img = img_container["img"]
            # if img is None:
            #     continue
            # print(img)

    else:
        st.write(pd.read_csv("Attendance/all_subs.csv"))














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










#
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration
# import av
# import cv2
# import streamlit as st
# import os
# import pandas as pd
# import datetime
# import time
#
# # Create necessary directories if they don't exist
# os.makedirs("ImagesUnknown", exist_ok=True)
# os.makedirs("Attendance", exist_ok=True)
#
# # Load face recognizer and other resources
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("TrainingImageLabel/Trainner.yml")
# cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Load employee details
# df = pd.read_csv("EmployeeDetails/EmployeeDetails.csv")
#
# # Attendance dataframe
# col_names = ['Id', 'Name', 'Date', 'Time']
# attendance = pd.DataFrame(columns=col_names)
#
# # Initialize Streamlit
# st.title("MVC Facial Recognition - Attendance Tracking")
# st.info("Testing WEB-RTC at client side")
#
# with st.expander("HOW TO USE"):
#     st.write("Click on start")
#     st.write("If asked for permissions, accept")
#     st.write("The app will start detecting and recognizing faces for attendance tracking")
#     st.write("Press 'q' to stop capturing and save the attendance records")
#
# # Define the video frame callback
# def video_frame_callback(frame):
#     global attendance
#
#     frm = frame.to_ndarray(format="bgr24")
#     gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
#     faces = cascade.detectMultiScale(gray, 1.2, 5)
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frm, (x, y), (x + w, y + h), (225, 0, 0), 2)
#         Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
#         if conf < 50:
#             ts = time.time()
#             date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#             timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#             aa = df.loc[df['Id'] == Id]['Name'].values[0]
#             tt = str(Id) + "-" + aa
#             attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
#         else:
#             Id = 'Unknown'
#             tt = str(Id)
#             if conf > 75:
#                 noOfFile = len(os.listdir("ImagesUnknown")) + 1
#                 cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", frm[y:y + h, x:x + w])
#         cv2.putText(frm, str(tt), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
#     return av.VideoFrame.from_ndarray(frm, format='bgr24')
#
# # Setup Streamlit WebRTC
# webrtc_streamer(
#     key="attendance",
#     video_frame_callback=video_frame_callback,
#     rtc_configuration=RTCConfiguration(
#         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#     )
# )
#
# # Save attendance records
# if st.button("Save Attendance"):
#     ts = time.time()
#     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#     Hour, Minute, Second = timeStamp.split(":")
#     fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
#     attendance.to_csv(fileName, index=False)
#     st.success(f"Attendance saved successfully to {fileName}")
#
# # Display current attendance dataframe
# st.dataframe(attendance)
