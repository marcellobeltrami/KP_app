import time
import speech_recognition as sr

def detect_speech_and_measure_time(mp3_file):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the mp3 file as the audio source
    with sr.AudioFile(mp3_file) as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready to detect speech. Processing audio...")

        # Start timing
        start_time = time.time()

        # Capture the audio
        audio = recognizer.record(source)

        # End timing
        end_time = time.time()

        # Calculate the duration
        duration = end_time - start_time

        try:
            # Recognize the speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            print(f"Detected Speech: {text}")
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")

        return float(duration)

# Example usage
if __name__ == "__main__":
    mp3_file = 'path/to/your/audio/file.mp3'  # Replace with your MP3 file
    detect_speech_and_measure_time(mp3_file)
