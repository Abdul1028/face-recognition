from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import streamlit as st

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def recieved( frame):
    frm = frame.to_ndarray(format="bgr24")

    faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

    for x, y, w, h in faces:
        cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return av.VideoFrame.from_ndarray(frm, format='bgr24')

st.title("MVC Facial recognition")
st.info("Testing WEB-RTC at client side")

with st.expander("HOW TO USE"):
    st.write("Click on start")
    st.write("if asked for permissions accept")
    st.write("App will start detecting your face and draw a rectangle over it")
    st.write("You can even change the camera and microphone input by clicking select device")

webrtc_streamer(key="key", video_frame_callback =recieved ,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)
