# Whisper Speech-to-Text with Diarization and RTTM Generation

This project implements a Streamlit-based web application that processes an uploaded audio file, enhances its quality, and performs speech-to-text transcription using OpenAI's Whisper model. Additionally, it applies speaker diarization using Pyannote and generates an RTTM (Rich Transcription Time Marked) file for speaker segmentation.

## Features
- Convert uploaded audio files to WAV format.
- Perform speaker diarization using Pyannote to distinguish between different speakers.
- Use Whisper for high-quality speech-to-text transcription.
- Split audio based on speaker segments and process each segment individually.
- Generate RTTM files for diarized speech.
- Plot original and enhanced audio signals. (still on dev phase)

## How it Works
1. **Audio Conversion:** Uploaded audio is converted to WAV format and limited to a maximum duration for faster processing.
2. **Audio Enhancement:** Noise reduction and normalization are applied to improve the audio quality.(still on dev phase)
3. **Speaker Diarization:** Pyannote's model identifies different speakers and generates speaker segments.
4. **Transcription:** The Whisper model transcribes each speaker segment, even splitting long chunks into smaller segments for accurate transcription.
5. **RTTM Generation:** An RTTM file is generated to mark speaker times and labels.

## Usage
TBC
