import streamlit as st
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import io
import numpy as np
import noisereduce as nr
from pyannote.audio import Pipeline  # For speaker diarization
import soundfile as sf  # Import soundfile to read from the buffer
import os
import librosa.display
import matplotlib.pyplot as plt
import tempfile
from pydub import AudioSegment
# Load Whisper model and processor
st.write("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
st.write("Whisper model loaded.")

# Load pyannote speaker diarization pipeline
st.write("Loading pyannote diarization model...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
st.write("pyannote diarization model loaded.")


# Step 1: Convert audio to WAV
@st.cache_data()
def convert_to_wav(file, max_duration=300):
    """Converts audio to WAV and limits the duration to max_duration seconds."""
    st.write("Step 1: Converting audio to WAV format...")
    audio_segment = AudioSegment.from_file(file)

    # Limit the audio duration to max_duration (e.g., 5 minutes = 300 seconds)
    if len(audio_segment) > max_duration * 1000:  # max_duration is in milliseconds
        audio_segment = audio_segment[:max_duration * 1000]
        st.write(f"Audio limited to {max_duration} seconds for faster processing.")

    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer


# load_audio_from_buffer to downsample audio
@st.cache_data()
def load_audio_from_buffer(buffer, target_sr=16000):
    st.write("Step 2: Loading and downsampling audio from buffer...")
    buffer.seek(0)
    audio, sr = sf.read(buffer)

    # If the sample rate is higher than the target, downsample
    if sr > target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        st.write(f"Audio downsampled to {sr} Hz.")

    return audio, sr


# Step 3: Enhance audio (normalize and reduce noise)
@st.cache_data()
def enhance_audio_and_plot(audio_buffer, target_sr=16000, chunk_size=30):
    """
    Enhances the audio quality by normalizing and reducing noise, then plots the original and enhanced audio.
    Also returns the enhanced audio as a .wav file.

    Parameters:
        audio_buffer: Audio buffer (uploaded audio file).
        target_sr (int): Target sample rate for downsampling (default is 16000 Hz).
        chunk_size (int): Size of each chunk for processing (in seconds).
    Returns:
        enhanced_audio_wav: Enhanced audio in .wav format as an in-memory buffer.
    """
    st.write("Enhancing audio...")

    try:
        # Read the audio file buffer into memory as a byte stream
        audio_buffer.seek(0)  # Ensure buffer pointer is at the start
        audio, sr = sf.read(io.BytesIO(audio_buffer.read()))  # Use soundfile to read the uploaded file

        # Downsample if necessary
        if sr > target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            st.write(f"Audio downsampled to {sr} Hz.")

        # Normalize the audio
        if np.max(np.abs(audio)) > 0:
            normalized_audio = audio / np.max(np.abs(audio))
        else:
            normalized_audio = audio  # Keep audio as is if max is zero (silent)

        # Noise reduction (process in chunks)
        chunk_length = chunk_size * sr
        enhanced_audio = np.array([])

        for i in range(0, len(normalized_audio), chunk_length):
            chunk = normalized_audio[i:i + chunk_length]
            reduced_noise_chunk = nr.reduce_noise(y=chunk, sr=sr)
            enhanced_audio = np.concatenate((enhanced_audio, reduced_noise_chunk))

        st.write(f"enhanced_audio  {enhanced_audio}.")
        # Plotting the original and enhanced audio signals
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        # Plot the original audio
        ax[0].set_title("Original Audio Signal")
        librosa.display.waveshow(audio, sr=sr, ax=ax[0], color='blue')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')

        # Plot the enhanced audio
        ax[1].set_title("Enhanced Audio Signal")
        librosa.display.waveshow(enhanced_audio, sr=sr, ax=ax[1], color='green')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Amplitude')

        # Show the plots
        st.pyplot(fig)

        # Save the enhanced audio to an in-memory buffer
        enhanced_audio_wav = io.BytesIO()
        sf.write(enhanced_audio_wav, enhanced_audio, sr, format='WAV')
        enhanced_audio_wav.seek(0)  # Reset buffer pointer

        return enhanced_audio_wav

    except EOFError:
        st.error("Error: The audio file appears to be incomplete or corrupted. Please upload a valid file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Step 4: Generate RTTM file using diarization
@st.cache_data()
def generate_rttm_file(buffer):
    st.write("Step 4: Generating RTTM file using pyannote speaker diarization...")

    # Save the buffer to a temporary WAV file
    temp_wav_file = "temp_audio.wav"
    buffer.seek(0)
    with open(temp_wav_file, 'wb') as f:
        f.write(buffer.read())

    # Perform speaker diarization
    diarization_result = pipeline({"audio": temp_wav_file})

    rttm_output = []

    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        rttm_output.append(
            f"SPEAKER unknown 1 {start_time:.3f} {end_time - start_time:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

    # Save RTTM to file
    rttm_filename = "output.rttm"
    with open(rttm_filename, 'w') as rttm_file:
        rttm_file.writelines(rttm_output)

    st.write(f"RTTM file saved as {rttm_filename}")

    # Clean up the temporary WAV file
    os.remove(temp_wav_file)

    return rttm_filename

@st.cache_data()
def save_audio_chunk(audio_data, sr, start_time, end_time, speaker_label, chunk_idx):
    # Create a unique filename for each chunk
    filename = f"temp_chunk_{speaker_label}_{chunk_idx}_{start_time:.2f}-{end_time:.2f}.wav"

    # Save the audio chunk in WAV format
    sf.write(filename, audio_data, sr)

    return filename

@st.cache_data()
def split_and_save_audio_chunks(wav_buffer, rttm_file):
    st.write("Step 5: Splitting audio into chunks based on RTTM file...")

    try:
        # Load audio using librosa (since it's better suited for in-memory data)
        wav_buffer.seek(0)  # Reset buffer position
        audio_data, sr = librosa.load(wav_buffer, sr=16000)  # Downsample if necessary
    except Exception as e:
        st.error(f"Failed to load audio data: {e}")
        return []

    # Assuming that `rttm_file` has the diarization info in text format
    # Parse the RTTM file and split audio accordingly
    chunks = []
    try:
        with open(rttm_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                parts = line.strip().split()
                start_time = float(parts[3])  # Start time in seconds
                duration = float(parts[4])  # Duration in seconds
                speaker_label = parts[7]  # Speaker label

                # Extract audio chunk
                start_sample = int(start_time * sr)
                end_sample = start_sample + int(duration * sr)
                chunk_audio = audio_data[start_sample:end_sample]

                # Save the chunk to a temporary WAV file
                chunk_filename = save_audio_chunk(chunk_audio, sr, start_time, start_time + duration, speaker_label, i)
                chunks.append((chunk_filename, start_time, start_time + duration, speaker_label))

    except Exception as e:
        st.error(f"Error processing RTTM file: {e}")
        return []

    st.write(f"Audio split into {len(chunks)} chunks based on RTTM file.")
    return chunks


# Step 6: Transcribe conversation for each speaker chunk with length check
@st.cache_data()
def transcribe_conversation_with_diarization(chunks, max_duration=30):
    st.write("Step 6: Transcribing conversation for each speaker...")

    conversation_transcript = []

    for i, (chunk_filename, start_time, end_time, speaker_label) in enumerate(chunks):
        st.write(f"Processing speaker {speaker_label}, from {start_time} to {end_time} seconds...")

        # Load the chunk audio file
        speaker_audio, sr = librosa.load(chunk_filename, sr=16000)  # Load chunk audio from the saved file

        # Check duration and split further if needed
        duration = librosa.get_duration(y=speaker_audio, sr=sr)  # Corrected function usage
        if duration > max_duration:
            st.write(f"Audio chunk too long ({duration:.2f}s). Splitting it into smaller chunks...")
            # Split audio chunk if too long
            split_audio_segments = librosa.effects.split(speaker_audio, top_db=30)
            for idx, (start, end) in enumerate(split_audio_segments):
                small_chunk_audio = speaker_audio[start:end]
                st.write(
                    f"Processing smaller chunk {idx + 1} for speaker {speaker_label}, from {start / sr:.2f} to {end / sr:.2f} seconds...")
                transcript = transcribe_audio_segment(small_chunk_audio, sr, speaker_label)
                conversation_transcript.append(f"Speaker {speaker_label}: {transcript}\n")
        else:
            # Process normally if duration is fine
            transcript = transcribe_audio_segment(speaker_audio, sr, speaker_label)
            conversation_transcript.append(f"Speaker {speaker_label}: {transcript}\n")

        # Clean up the temporary chunk file
        os.remove(chunk_filename)

    # Join the conversation transcript
    return "\n".join(conversation_transcript)


# Helper function for transcribing an audio segment
@st.cache_data()
def transcribe_audio_segment(audio_segment, sr, speaker_label):
    # Transcribe the audio chunk using Whisper
    input_features = processor(audio_segment, sampling_rate=sr, return_tensors="pt",language="ar")
    input_values = input_features.get("input_values") or input_features.get("input_features")

    if input_values is None:
        st.error(f"No valid input found for speaker {speaker_label}.")
        return ""

    with torch.inference_mode():
        # Generate transcript for the chunk
        output_ids = model.generate(
            input_values,
            max_length=512,
            num_beams=5,
            temperature=0.7,
            top_p=0.95,
            early_stopping=True
        )

    # Decode the predicted tokens
    transcript = processor.decode(output_ids[0], skip_special_tokens=True)
    return transcript


# Streamlit app
st.title("Whisper Speech-to-Text with Diarization and RTTM Generation")

# Upload an audio file for transcription
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'flac', 'm4a', 'ogg'])

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}, Size: {uploaded_file.size} bytes")

    with st.spinner('Processing...'):
        # Step 1: Convert to WAV
        wav_file = convert_to_wav(uploaded_file)

        # Step 2: Load audio from buffer
        audio, sr = load_audio_from_buffer(wav_file)

        # Step 3: Enhance the audio
       # with st.spinner("Processing and enhancing audio..."):
        #    enhanced_audio_wav = enhance_audio_and_plot(wav_file)
         #   if enhanced_audio_wav:
          #      st.success("Audio processing completed.")

        # Step 4: Generate RTTM file
        rttm_file = generate_rttm_file(wav_file)

        # Step 5: Split audio into chunks
        chunks = split_and_save_audio_chunks(wav_file, rttm_file)

        # Step 6: Transcribe and diarize conversation
        result = transcribe_conversation_with_diarization(chunks)

        st.success("Audio processing completed.")
        st.write(result)