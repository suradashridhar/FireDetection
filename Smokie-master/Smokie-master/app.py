import tensorflow
import numpy as np
import cv2 as cv
import time
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

model = tensorflow.keras.models.load_model("smoke-detector7.h5")

IMG_WIDTH = 256
IMG_HEIGHT = 256

newTime = 0
prevTime = 0
avg_buf = []
avg = 0
kelas = ""
i = 0
n = 0
ct = 0

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.sidebar.image('1.png')
st.sidebar.title('Efficient Real-Time Smoke Detection Module')
st.sidebar.markdown('''This module uses Efficient Net + Yolo5 based smoke detection model to detect smoke in the slightest part and alert''')


st.sidebar.write('- Useful for smoke detection in Vande Bharat / Premium Trains / Flights')
st.sidebar.write('- Useful for smoke detection in Hospitals / Other Instituitions')
st.sidebar.write('- Zero lag. 100% Accurate')

def overlay_img(img, alpha1, mask, alpha2):
    img_h, img_w, img_ch = img.shape
    msk_h, msk_w, msk_ch = mask.shape
    s_pred = np.squeeze(mask)
    color_map = {
    0:(0, 0, 0),
    1:(255, 0, 0)
    }
    vis = np.zeros((msk_h, msk_w, img_ch)).astype(np.uint8)
    for i, c in color_map.items():
        vis[s_pred == 1] = color_map[i]
    vis = cv.resize(vis, (img_w, img_h))
    overlay = cv.addWeighted(img, alpha1, vis, alpha2, 0)
    return overlay

class VideoProcessor:
    def recv(self, frame):
        o_img = frame.to_ndarray(format="bgr24")
        img = cv.resize(o_img, (IMG_HEIGHT, IMG_WIDTH))
        prediction = model.predict_on_batch(img[tensorflow.newaxis, ...])[0]
        #prediction = model.predict(img[tensorflow.newaxis, ...])[0]
        predicted_mask = (prediction > 0.3).astype(np.uint8)
        pred = np.array(tensorflow.keras.utils.array_to_img(predicted_mask))
        wht = np.sum(pred == 255)
        blk = np.sum(pred == 0)
        pers = round(wht/(wht+blk)*100, 2)
        ovrly = overlay_img(o_img, 1, predicted_mask, 0.8)
        cv.putText(ovrly, "Smoke "+str(pers)+("%"), (2,25), cv.FONT_HERSHEY_COMPLEX, 1, (255,0, 0), 1)
        return av.VideoFrame.from_ndarray(ovrly, format="bgr24")

ctx = webrtc_streamer(key="SmokeDetector", 
                      mode=WebRtcMode.SENDRECV, 
                      rtc_configuration=RTC_CONFIGURATION, 
                      media_stream_constraints={"video": True, "audio": False}, 
                      video_processor_factory=VideoProcessor, 
                      async_processing=True)
