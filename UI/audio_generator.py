# audio_generator.py
import numpy as np
from scipy.io.wavfile import write
import os

def generate_audio(random_seed):
    np.random.seed(random_seed)
    # Generate a simple sine wave with random frequency
    sample_rate = 44100
    duration = 3  # seconds
    frequency = np.random.randint(220, 880)  # random frequency between A3 and A5
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Normalize
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save to file
    output_path = f"generated_audio_{random_seed}.wav"
    write(output_path, sample_rate, audio_data)
    return output_path