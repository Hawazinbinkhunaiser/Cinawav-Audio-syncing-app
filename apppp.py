pip install streamlit-audio-recorder

import streamlit as st
from streamlit_audio_recorder import audio_recorder

# Title
st.title("Audio Recorder Demo")

# Record Audio
audio_bytes = audio_recorder()

# Play back the recorded audio
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
        st.success("Audio saved as output.wav")
