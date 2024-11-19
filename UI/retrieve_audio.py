# app.py
import streamlit as st
from audio_generator import generate_audio

st.title("Audio Generator")

# User input for seed number
seed_number = st.number_input("Enter a number (1-10000)", 
                            min_value=1, 
                            max_value=10000, 
                            value=1,
                            step=1)

if st.button("Generate Audio"):
    # Generate audio file
    audio_path = generate_audio(seed_number)
    
    # Display the seed
    st.write(f"Generated audio with seed: {seed_number}")
    
    # Play the audio
    with open(audio_path, "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/wav')
