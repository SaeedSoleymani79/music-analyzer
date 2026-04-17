import streamlit as st
import librosa
import numpy as np

# Standard Major/Minor profiles (Krumhansl-Kessler)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def detect_key(y, sr):
    # Get the chromagram (energy of each note)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Check correlation against Major and Minor templates
    best_key = ""
    best_corr = -1
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for i in range(12):
        # Rotate the profile to check each starting note
        rotated_major = np.roll(MAJOR_PROFILE, i)
        rotated_minor = np.roll(MINOR_PROFILE, i)
        
        major_corr = np.corrcoef(chroma_mean, rotated_major)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, rotated_minor)[0, 1]
        
        if major_corr > best_corr:
            best_corr, best_key = major_corr, f"{notes[i]} Major"
        if minor_corr > best_corr:
            best_corr, best_key = minor_corr, f"{notes[i]} Minor"
            
    return best_key

st.title("🎵 Online Music Analyzer")

uploaded_file = st.file_uploader("Upload Audio", type=['mp3', 'wav'])

if uploaded_file:
    with st.spinner("Analyzing..."):
        y, sr = librosa.load(uploaded_file)
        
        # 1. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 2. Key/Scale
        key = detect_key(y, sr)
        
        col1, col2 = st.columns(2)
        col1.metric("Tempo", f"{round(float(tempo))} BPM")
        col2.metric("Detected Key", key)
        
        # 3. Visualizer
        st.write("### Pitch Distribution (Chromagram)")
        st.bar_chart(np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1))
