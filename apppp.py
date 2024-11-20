
import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Audio Recorder with Streamlit")

webrtc_streamer(key="audio-recorder", media_stream_constraints={"audio": True})
