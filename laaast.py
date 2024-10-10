import streamlit as st
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from docx import Document
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import librosa.display
import tempfile
from docx.shared import Inches
import subprocess
import os

# Function definitions (find_offset, process_segment_data, generate_docx, format_time, detect_dropouts, plot_waveform_with_dropouts)
# ... (keep these functions as they were in your original code)

# Streamlit UI
st.title("7.1 Audio Channel Extractor and Sync Analyzer")
st.write("Upload your M4A file, and I'll extract each audio channel, analyze sync, and detect dropouts.")

# File upload
uploaded_file = st.file_uploader("Choose an M4A file", type=["m4a"])

# Channel Mapping for 7.1 Surround Sound
channels = {
    'Front Left (FL)': 'FL',
    'Front Right (FR)': 'FR',
    'Center (FC)': 'FC',
    'Subwoofer (LFE)': 'LFE',
    'Back Left (BL)': 'BL',
    'Back Right (BR)': 'BR',
    'Side Left (SL)': 'SL',
    'Side Right (SR)': 'SR'
}

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("input_file.m4a", "wb") as f:
        f.write(uploaded_file.getbuffer())
    input_file = "input_file.m4a"

    st.write("File uploaded successfully. Extracting audio channels...")

    # Extract each channel using FFmpeg pan filter
    output_files = []
    for name, channel in channels.items():
        output_file = f"{name.replace(' ', '_').lower()}.wav"
        command = f'ffmpeg -i "{input_file}" -filter_complex "pan=mono|c0={channel}" "{output_file}"'
        subprocess.run(command, shell=True)
        output_files.append(output_file)
        st.write(f"Extracted {name} to {output_file}")

    st.write("Extraction complete. Proceeding with sync analysis and dropout detection.")

    # Set center channel as master file
    master_file = "center_(fc).wav"

    # Sampling rate and segment settings
    low_sr = st.slider("Select lower sampling rate for faster processing", 4000, 16000, 4000)
    segment_length = st.slider("Segment length (seconds)", 2, 120, 10)
    intervals = st.multiselect("Select intervals (in seconds)", options=[60, 900, 1800, 2700, 3600, 4500, 5400, 6300], default=[60, 900, 1800, 2700, 3600])

    # Add a "Process" button
    if st.button("Process"):
        st.write("Processing started...")

        # Load the master track
        master, sr_master = librosa.load(master_file, sr=low_sr)

        all_results = {}
        all_dropouts = {}
        all_plots = {}

        for sample_file in output_files:
            if sample_file != master_file:
                sample, sr_sample = librosa.load(sample_file, sr=low_sr)

                # Resample if the sampling rates do not match
                if sr_master != sr_sample:
                    sample = librosa.resample(sample, sr_sample, sr_master)

                args = [(interval, master, sample, sr_master, segment_length) for interval in intervals]

                with ProcessPoolExecutor() as executor:
                    results = list(filter(None, executor.map(process_segment_data, args)))

                all_results[sample_file] = results

                # Detect dropouts in the sample track
                dropouts = detect_dropouts(sample_file)
                all_dropouts[sample_file] = dropouts

                # Plot waveform with dropouts
                plot_file_name = f"{sample_file}_plot.png"
                plot_waveform_with_dropouts(sample, sr_master, dropouts, plot_file_name)
                all_plots[sample_file] = plot_file_name

        st.write("Processing completed.")
        
        # Display results
        for sample_name, results in all_results.items():
            st.subheader(f"Results for {sample_name}:")
            for interval, offset in results:
                st.write(f"At {interval} mins: Offset = {offset:.2f} ms")
        
        # Display dropouts
        for sample_name, dropouts in all_dropouts.items():
            st.subheader(f"Detected Dropouts for {sample_name}:")
            if dropouts:
                for dropout in dropouts:
                    start, end, duration_ms = dropout
                    st.write(f"Start: {format_time(start)} | End: {format_time(end)} | Duration: {duration_ms:.0f} ms")
            else:
                st.write("No significant dropouts detected.")

        # Generate DOCX and provide download option
        doc = generate_docx(all_results, intervals, all_dropouts, all_plots)
        doc_buffer = BytesIO()
        doc.save(doc_buffer)
        doc_buffer.seek(0)
        st.download_button("Download Results as DOCX", data=doc_buffer.getvalue(), file_name="audio_sync_results.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    # Clean up temporary files
    for file in output_files + ["input_file.m4a"]:
        os.remove(file)
else:
    st.warning("Please upload an M4A file to begin processing.")
