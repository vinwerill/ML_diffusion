import streamlit as st

options = ['紅嘴黑鵯', '白頭翁']
image_paths = {
    '紅嘴黑鵯': './picture/blackbulbul.jpeg',
    '白頭翁': './picture/white.jpeg',
}
audio_paths = {
    '紅嘴黑鵯': './black.mp3',
    '白頭翁': './white.mp3',
}

selected_option = st.radio("Choose one bird species and generate its sound", options)

# Display the corresponding image and audio
if selected_option:
    st.image(image_paths[selected_option], caption=selected_option)
    audio_file_path = audio_paths[selected_option]
    with open(audio_file_path, "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/mp3')

# Create one-hot vector
one_hot_vector = [1 if option == selected_option else 0 for option in options]

# Add number input fields
seed = st.number_input("Enter seed number", value=0)

# Add calculate button
if st.button(f"Generate {selected_option} sound"):
    st.write(f"The sound is: {selected_option}")
    with open(audio_file_path, "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/mp3')
        