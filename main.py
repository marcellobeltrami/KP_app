import argparse

parser = argparse.ArgumentParser(description="My friends yap. We need to know who deserved the Yappology degree the most.")

# Add commands and arguments to the parser
subparsers = parser.add_subparsers(title="commands", dest="command")

# Command: greet
mp3_file = subparsers.add_parser("input", help="mp3 file of a conversation")

audio_split(mp3_file, out_dir="./temp/")


from tensorflow.keras.models import load_model
from scripts.timing import detect_speech_and_measure_time
from scripts.diarization import audio_split

# Path to your saved model
model_path = './data/model/saved_model.h5' 

# Load the model
model = load_model(model_path)


import librosa
import numpy as np
import os

# Example function to preprocess new audio data
def preprocess_audio(audio_file):
    # Load audio using librosa
    audio, sr = librosa.load(audio_file, sr=16000)  # Adjust sr as per your model's requirement

    # Extract features (e.g., spectrogram)
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    log_S = librosa.power_to_db(S, ref=np.max)

    # Reshape to match model input shape (if necessary)
    # log_S = np.expand_dims(log_S, axis=-1)  # Example if input shape is (time_steps, num_features, 1)

    return log_S

# Example usage to preprocess new audio file
files = os.listdir("./temp/")

    # Filter files with .mp3 extension
audio_files = [file for file in files if file.endswith('.mp3')]



yapping_time = {}

for audio in audio_files:
    
    new_input = preprocess_audio(audio)

    # Example: Predicting using the loaded model
    predictions = model.predict(np.expand_dims(new_input, axis=0))  # Add batch dimension if necessary

    # Example: Get predicted class
    predicted_class = np.argmax(predictions, axis=-1)[0]
    print(f"{predicted_class}")

    # Store yapping time
    if predicted_class not in yapping_time:
        yapping_time[predicted_class] = detect_speech_and_measure_time(audio)
    else:
        yapping_time[predicted_class] += detect_speech_and_measure_time(audio)

# Print total yapping time per class
for class_label, total_time in yapping_time.items():
    print(f"Class {class_label} total yapping time: {total_time/60} minutes")

