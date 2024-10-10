import streamlit as st
import librosa
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor
from docx import Document
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import librosa.display
import tempfile
from docx.shared import Inches

# Function to find the offset between master and sample segments
def find_offset(master_segment, sample_segment, sr):
    correlation = np.correlate(master_segment, sample_segment, mode='full')
    max_corr_index = np.argmax(correlation)
    offset_samples = max_corr_index - len(sample_segment) + 1
    offset_ms = (offset_samples / sr) * 1000  # Convert to milliseconds
    return offset_ms

# Process segment data function for parallel execution
def process_segment_data(args):
    interval, master, sample, sr_master, segment_length = args
    start = interval * sr_master
    end = start + segment_length * sr_master  # Segment of defined length
    if end <= len(master) and end <= len(sample):
        master_segment = master[start:end]
        sample_segment = sample[start:end]
        offset = find_offset(master_segment, sample_segment, sr_master)
        return (interval // 60, offset)
    return None

# Function to detect dropouts in an audio file
def detect_dropouts(file_path, dropout_db_threshold=-20, min_duration_ms=100):
    y, sr = librosa.load(file_path, sr=None)
    hop_length = 256
    frame_length = hop_length / sr * 1000
    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)
    rms_db = librosa.power_to_db(rms, ref=np.max)
    dropout_frames = rms_db[0] < dropout_db_threshold

    min_frames = int(min_duration_ms / frame_length)
    dropouts = []
    start = None

    for i, is_dropout in enumerate(dropout_frames):
        if is_dropout and start is None:
            start = i  # Start of a dropout
        elif not is_dropout and start is not None:
            if i - start >= min_frames:
                start_time = start * hop_length / sr
                end_time = i * hop_length / sr
                duration_ms = (end_time - start_time) * 1000
                dropouts.append((start_time, end_time, duration_ms))
            start = None

    return dropouts

# Function to plot waveform with dropouts
def plot_waveform_with_dropouts(y, sr, dropouts, file_name):
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title('Waveform with Detected Dropouts')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    for dropout in dropouts:
        start_time, end_time, _ = dropout
        plt.axvspan(start_time, end_time, color='red', alpha=0.5, label='Dropout' if 'Dropout' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# Streamlit UI
st.title("Audio Sync and Dropout Detection with M4A Extraction")

# Upload M4A file
m4a_file = st.file_uploader("Upload M4A file", type=["m4a"])

# Channel mapping for 7.1 surround sound
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

if m4a_file:
    st.write("File uploaded. Extracting channels...")

    # Save M4A file to temp location
    with open("input_file.m4a", "wb") as f:
        f.write(m4a_file.getbuffer())
    input_file = "input_file.m4a"
    
    # Extract each channel using FFmpeg
    output_files = []
    for name, channel in channels.items():
        output_file = f"{name.replace(' ', '_').lower()}.wav"
        command = f'ffmpeg -i "{input_file}" -filter_complex "pan=mono|c0={channel}" "{output_file}"'
        subprocess.run(command, shell=True)
        output_files.append(output_file)
        st.write(f"Extracted {name} to {output_file}")

    # Assign the third channel (Center) as the master track
    master_track = output_files[2]
    st.write(f"Center channel ({output_files[2]}) selected as master track.")

    # Proceed to sync and dropout detection using existing logic
    low_sr = st.slider("Select lower sampling rate for faster processing", 4000, 16000, 4000)
    segment_length = st.slider("Segment length (seconds)", 2, 120, 10)
    intervals = st.multiselect("Select intervals (in seconds)", options=[60, 900, 1800, 2700, 3600, 4500, 5400, 6300], default=[60, 900, 1800, 2700, 3600])

    if st.button("Process"):
        st.write("Processing started...")

        # Load the master track
        master, sr_master = librosa.load(master_track, sr=low_sr)
        
        all_results = {}
        all_dropouts = {}
        all_plots = {}

        # Process each sample track
        for sample_file in output_files:
            if sample_file != master_track:
                sample, sr_sample = librosa.load(sample_file, sr=low_sr)

                if sr_master != sr_sample:
                    sample = librosa.resample(sample, sr_sample, sr_master)

                args = [(interval, master, sample, sr_master, segment_length) for interval in intervals]

                with ProcessPoolExecutor() as executor:
                    results = list(filter(None, executor.map(process_segment_data, args)))

                all_results[sample_file] = results

                dropouts = detect_dropouts(sample_file)
                all_dropouts[sample_file] = dropouts

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
                    st.write(f"Start: {start:.2f} secs | End: {end:.2f} secs | Duration: {duration_ms:.0f} ms")
            else:
                st.write("No significant dropouts detected.")
