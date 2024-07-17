from pydub import AudioSegment
from pyannote.audio import Pipeline
import os

def audio_split(audio_path,out_dir:str):
    # Load the pre-trained pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # Perform speaker diarization
    diarization = pipeline(audio_path)

    # Load the audio file using pydub
    audio = AudioSegment.from_mp3(audio_path)

    # Create a directory to save the split audio files
    output_dir = out_dir
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the speaker segments and save them
    for i, turn in enumerate(diarization.itertracks(yield_label=True)):
        start_time = turn[0].start * 1000  # Convert to milliseconds
        end_time = turn[0].end * 1000  # Convert to milliseconds
        speaker_label = turn[2]
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"{i}_{speaker_label}.mp3"), format="mp3")

    print(f"Segments saved in {output_dir}")
