# import csv
# import os
# import datetime
# import time
# import cv2
# import numpy as np
# import pandas as pd
# import streamlit as st
# from PIL import Image
#
# def TakeImages(Id,name):
#     # Id = (txt.get())
#     # name = (txt2.get())
#     if(Id and name):
#         cam = cv2.VideoCapture(0)
#         #url= 'http://192.168.43.73:8080/shot.jpg'
#         harcascadePath = "haarcascade_frontalface_default.xml"
#         detector = cv2.CascadeClassifier(harcascadePath)
#         sampleNum = 0
#         while(True):
#             ret, img = cam.read()
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = detector.detectMultiScale(gray, 1.3, 5)
#             # gcontext = SSLContext(PROTOCOL_TLSv1)  # Only for gangstars
#             #info = urlopen(url, context=gcontext).read()
#
#             # imgNp=np.array(bytearray(info),dtype=np.uint8)
#             # image_frame=cv2.imdecode(imgNp,-1)
#
#             # Convert frame to grayscale
#             #gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
#
#             # Detect frames of different sizes, list of faces rectangles
#             #faces = face_detector.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                 # incrementing sample number
#                 sampleNum = sampleNum+1
#                 # saving the captured face in the dataset folder TrainingImage
#                 cv2.imwrite("TrainingImage\\"+name + "."+Id + '.' +
#                             str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
#                 # display the frame
#                 print("created")
#                 cv2.imshow('frame', img)
#             # wait for 100 miliseconds
#             if cv2.waitKey(100) & 0xFF == ord('q'):
#                 break
#             # break if the sample number is morethan 100
#             elif sampleNum > 100:
#                 break
#         cam.release()
#         cv2.destroyAllWindows()
#         res = "Images Saved for ID : " + Id + " Name : " + name
#         row = [Id, name]
#         with open('EmployeeDetails\\EmployeeDetails.csv', 'a+') as csvFile:
#             writer = csv.writer(csvFile)
#             writer.writerow(row)
#         csvFile.close()
#         st.toast(res)
#     else:
#         if(name.isalpha()):
#             res = "Enter Numeric Id"
#             st.toast(res)
#
#
# def TrainImages():
#     recognizer = cv2.face_LBPHFaceRecognizer.create()
#     #recognizer = cv2.face.LBPHFaceRecognizer_create()
#     # recognizer=cv2.createLBPHFaceRecognizer()
#     # recognizer=cv2.face.EigenFaceRecognizer_create()
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     detector = cv2.CascadeClassifier(harcascadePath)
#     faces, Id = getImagesAndLabels("TrainingImage")
#     recognizer.train(faces, np.array(Id))
#     recognizer.save("TrainingImageLabel\\Trainner.yml")
#     res = "Image Trained"  # +",".join(str(f) for f in Id)
#     st.toast(res)
#
#
# def getImagesAndLabels(path):
#     # get the path of all the files in the folder
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     # print(imagePaths)
#
#     # create empth face list
#     faces = []
#     # create empty ID list
#     Ids = []
#     # now looping through all the image paths and loading the Ids and the images
#     for imagePath in imagePaths:
#         # loading the image and converting it to gray scale
#         pilImage = Image.open(imagePath).convert('L')
#         # Now we are converting the PIL image into numpy array
#         imageNp = np.array(pilImage, 'uint8')
#         # getting the Id from the image
#         Id = int(os.path.split(imagePath)[-1].split(".")[1])
#         # extract the face from the training image sample
#         faces.append(imageNp)
#         Ids.append(Id)
#     return faces, Ids
#
#
# def TrackImages():
#     #recognizer = cv2.face.LBPHFaceRecognizer_create()
#     # recognizer=cv2.createLBPHFaceRecognizer()
#     recognizer = cv2.face_LBPHFaceRecognizer.create()
#     recognizer.read("TrainingImageLabel\\Trainner.yml")
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(harcascadePath)
#     df = pd.read_csv("EmployeeDetails\\EmployeeDetails.csv")
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     col_names = ['Id', 'Name', 'Date', 'Time']
#     attendance = pd.DataFrame(columns=col_names)
#     while True:
#         ret, im = cam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.2, 5)
#         for(x, y, w, h) in faces:
#             cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
#             Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
#             if(conf < 50):
#                 ts = time.time()
#                 date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 timeStamp = datetime.datetime.fromtimestamp(
#                     ts).strftime('%H:%M:%S')
#                 aa = df.loc[df['Id'] == Id]['Name'].values
#                 tt = str(Id) + "-" + aa
#                 attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
#
#             else:
#                 Id = 'Unknown'
#                 tt = str(Id)
#             if(conf > 75):
#                 noOfFile = len(os.listdir("ImagesUnknown"))+1
#                 cv2.imwrite("ImagesUnknown\\Image"+str(noOfFile) +
#                             ".jpg", im[y:y+h, x:x+w])
#             cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)
#         attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
#         cv2.imshow('im', im)
#         # if (cv2.waitKey(10000)):
#         #  break
#         if (cv2.waitKey(1) == ord('q')):
#             break
#     ts = time.time()
#     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#     Hour, Minute, Second = timeStamp.split(":")
#     fileName = "Attendance\\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
#     attendance.to_csv(fileName, index=False)
#     cam.release()
#     # cv2.waitKey
#     cv2.destroyAllWindows()
#     # print(attendance)
#     res = attendance
#     st.toast(res)
#
# def main():
#     st.title("Student Attendance System")
#
#     # Buttons to choose functionalities
#     choice = st.sidebar.selectbox("Choose Functionality", ["Take Image", "Train Image", "Track Attendance"])
#
#     haarcasecade_path = "haarcascade_frontalface_default.xml"
#     trainimage_path = "TrainingImage"
#     trainimagelabel_path = "TrainingImageLabel/trainner.yml"
#
#     if choice == "Take Image":
#         st.header("Take Image")
#         l1 = st.text_input("Enrollment Number")
#         l2 = st.text_input("Name")
#         if st.button("Capture Images"):
#             TakeImages(l1, l2)
#
#     elif choice == "Train Image":
#         st.header("Train Image")
#         if st.button("Train Images"):
#             TrainImages()
#
#     elif choice == "Track Attendance":
#         st.header("Track Attendance")
#         TrackImages()
#
#
# if __name__ == "__main__":
#     main()
#
# import csv
# import os
# import datetime
# import time
# import cv2
# import numpy as np
# import pandas as pd
# import streamlit as st
# from PIL import Image
#
# def TakeImages(Id, name):
#     if(Id and name):
#         cam = cv2.VideoCapture(0)
#         harcascadePath = "haarcascade_frontalface_default.xml"
#         detector = cv2.CascadeClassifier(harcascadePath)
#         sampleNum = 0
#         image_placeholder = st.empty()
#         while(True):
#             ret, img = cam.read()
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = detector.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                 sampleNum += 1
#                 cv2.imwrite("TrainingImage\\"+name + "."+Id + '.' + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
#             image_placeholder.image(img, channels="BGR")
#             if sampleNum > 100:
#                 break
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         cam.release()
#         cv2.destroyAllWindows()
#         res = "Images Saved for ID : " + Id + " Name : " + name
#         row = [Id, name]
#         with open('EmployeeDetails\\EmployeeDetails.csv', 'a+') as csvFile:
#             writer = csv.writer(csvFile)
#             writer.writerow(row)
#         st.success(res)
#     else:
#         if name.isalpha():
#             st.error("Enter Numeric Id")
#
# def TrainImages():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     detector = cv2.CascadeClassifier(harcascadePath)
#     faces, Id = getImagesAndLabels("TrainingImage")
#     recognizer.train(faces, np.array(Id))
#     recognizer.save("TrainingImageLabel\\Trainner.yml")
#     st.success("Images Trained")
#
# def getImagesAndLabels(path):
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces = []
#     Ids = []
#     for imagePath in imagePaths:
#         pilImage = Image.open(imagePath).convert('L')
#         imageNp = np.array(pilImage, 'uint8')
#         Id = int(os.path.split(imagePath)[-1].split(".")[1])
#         faces.append(imageNp)
#         Ids.append(Id)
#     return faces, Ids
#
# def TrackImages():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("TrainingImageLabel\\Trainner.yml")
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(harcascadePath)
#     df = pd.read_csv("EmployeeDetails\\EmployeeDetails.csv")
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     col_names = ['Id', 'Name', 'Date', 'Time']
#     attendance = pd.DataFrame(columns=col_names)
#     image_placeholder = st.empty()
#     while True:
#         ret, im = cam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.2, 5)
#         for(x, y, w, h) in faces:
#             cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
#             Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
#             if(conf < 50):
#                 ts = time.time()
#                 date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                 aa = df.loc[df['Id'] == Id]['Name'].values[0]
#                 tt = str(Id) + "-" + aa
#                 attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
#             else:
#                 Id = 'Unknown'
#                 tt = str(Id)
#             if(conf > 75):
#                 noOfFile = len(os.listdir("ImagesUnknown"))+1
#                 cv2.imwrite("ImagesUnknown\\Image"+str(noOfFile) + ".jpg", im[y:y+h, x:x+w])
#             cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)
#         attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
#         image_placeholder.image(im, channels="BGR")
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     ts = time.time()
#     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#     Hour, Minute, Second = timeStamp.split(":")
#     fileName = "Attendance\\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
#     attendance.to_csv(fileName, index=False)
#     cam.release()
#     cv2.destroyAllWindows()
#     st.success("Attendance Tracked")
#     st.dataframe(attendance)
#
# def main():
#     st.title("Student Attendance System")
#
#     choice = st.sidebar.selectbox("Choose Functionality", ["Take Image", "Train Image", "Track Attendance"])
#
#     if choice == "Take Image":
#         st.header("Take Image")
#         l1 = st.text_input("Enrollment Number")
#         l2 = st.text_input("Name")
#         if st.button("Capture Images"):
#             TakeImages(l1, l2)
#
#     elif choice == "Train Image":
#         st.header("Train Image")
#         if st.button("Train Images"):
#             TrainImages()
#
#     elif choice == "Track Attendance":
#         st.header("Track Attendance")
#         TrackImages()
#
# if __name__ == "__main__":
#     main()


import csv
import os
import datetime
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase,RTCConfiguration
def save_uploaded_file(uploaded_file, folder):
    try:
        with open(os.path.join(folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False


class FaceDetection():
    def __init__(self):
        self.harcascadePath = "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(self.harcascadePath)
        self.sampleNum = 0
        self.Id = None
        self.name = None
        self.folder = "TrainingImage"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.sampleNum += 1
            cv2.imwrite(os.path.join(self.folder, f"{self.name}.{self.Id}.{self.sampleNum}.jpg"),
                        gray[y:y + h, x:x + w])
            # if self.sampleNum > 100:
            #     break
        return img


def take_images(Id, name):
    if Id and name:
        st.info("Capturing images from webcam")
        transformer = FaceDetection()
        transformer.Id = Id
        transformer.name = name
        webrtc_streamer(key="key", video_processor_factory=FaceDetection,
                        rtc_configuration=RTCConfiguration(
                            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                        )
                        )

        res = f"Images Saved for ID : {Id} Name : {name}"
        row = [Id, name]
        with open('EmployeeDetails/EmployeeDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        st.success(res)
    else:
        if name.isalpha():
            st.error("Enter Numeric Id")


def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = get_images_and_labels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    st.success("Images Trained")


def get_images_and_labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def track_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("EmployeeDetails/EmployeeDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    image_placeholder = st.empty()

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values[0]
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 75:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        image_placeholder.image(im, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    st.success("Attendance Tracked")
    st.dataframe(attendance)


def main():
    st.title("Student Attendance System")

    choice = st.sidebar.selectbox("Choose Functionality", ["Take Image", "Train Image", "Track Attendance"])

    if choice == "Take Image":
        st.header("Take Image")
        l1 = st.text_input("Enrollment Number")
        l2 = st.text_input("Name")
        if st.button("Capture Images"):
            take_images(l1, l2)

    elif choice == "Train Image":
        st.header("Train Image")
        if st.button("Train Images"):
            train_images()

    elif choice == "Track Attendance":
        st.header("Track Attendance")
        track_images()


if __name__ == "__main__":
    main()
