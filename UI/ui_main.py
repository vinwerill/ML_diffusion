import streamlit as st
import os
from audio_generator import generate_audio

# Define constants
options = ['紅嘴黑鵯', '白頭翁', '大卷尾', '樹鵲', '綠繡眼', '五色鳥', '麻雀']
ch_to_en = {
    "紅嘴黑鵯": "black",
    "白頭翁": "white",
    "大卷尾": "bladro",
    "樹鵲": "tree",
    "綠繡眼": "green",
    "五色鳥": "color",
    "麻雀": "sparrow"
}

# Use absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
image_paths = {
    '紅嘴黑鵯': os.path.join(base_dir, 'picture/blackbulbul.jpeg'),
    '白頭翁': os.path.join(base_dir, 'picture/white.jpeg'),
    '大卷尾': os.path.join(base_dir, 'picture/bladro.jpeg'),
    '樹鵲': os.path.join(base_dir, 'picture/tree.jpeg'),
    '綠繡眼': os.path.join(base_dir, 'picture/green.jpeg'),
    '五色鳥': os.path.join(base_dir, 'picture/color.jpeg'),
    '麻雀': os.path.join(base_dir, 'picture/sparrow.jpeg')
}
audio_paths = {
    '紅嘴黑鵯': os.path.join(base_dir, 'sample_audio/black.mp3'),
    '白頭翁': os.path.join(base_dir, 'sample_audio/white.mp3'),
    '大卷尾': os.path.join(base_dir, 'sample_audio/bladro.mp3'),
    '樹鵲': os.path.join(base_dir, 'sample_audio/tree.mp3'),
    '綠繡眼': os.path.join(base_dir, 'sample_audio/green.mp3'),
    '五色鳥': os.path.join(base_dir, 'sample_audio/color.mp3'),
    '麻雀': os.path.join(base_dir, 'sample_audio/sparrow.mp3')
}

selected_option = st.radio("Choose one bird species and generate its sound", options)

# Display the corresponding image and audio
if selected_option:
    st.image(image_paths[selected_option], caption=selected_option)
    audio_file_path = audio_paths[selected_option]
    with open(audio_file_path, "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/mp3')

# Add number input fields
seed = st.number_input("Enter seed number", value=0)

ckpt_name = ch_to_en[selected_option]

# Add calculate button
if st.button(f"Generate {selected_option} sound"):
    st.write(f"Generating {selected_option} sound with seed: {seed}")
    generate_audio(ckpt_name, seed)
    st.write(f"The sound is: {selected_option}")
    with open("generated.wav", "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/wav')