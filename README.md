# Whisper Speech-to-Text with Diarization and RTTM Generation

This project implements a Streamlit-based web application that processes an uploaded audio file, enhances its quality, and performs speech-to-text transcription using OpenAI's Whisper model. Additionally, it applies speaker diarization using Pyannote and generates an RTTM (Rich Transcription Time Marked) file for speaker segmentation.

## Features
- Convert uploaded audio files to WAV format.
- Perform speaker diarization using [Pyannote Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) to distinguish between different speakers.
- Use [Whisper](https://huggingface.co/openai/whisper-large-v3) for high-quality speech-to-text transcription.
- Split audio based on speaker segments and process each segment individually.
- Generate RTTM files for diarized speech.
- Plot original and enhanced audio signals.

## How it Works
1. **Audio Conversion:** Uploaded audio is converted to WAV format and limited to a maximum duration for faster processing.
2. **Audio Enhancement:** Noise reduction and normalization are applied to improve the audio quality.
3. **Speaker Diarization:** Pyannote's [Speaker Diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1) identifies different speakers and generates speaker segments.
4. **Transcription:** The [Whisper model](https://huggingface.co/openai/whisper-large-v3) transcribes each speaker segment, even splitting long chunks into smaller segments for accurate transcription.
5. **RTTM Generation:** An RTTM file is generated to mark speaker times and labels.

##Next Features

**The following features are planned for future updates:**

1.	**Refine Output for Typo Corrections Using LLaMA 3.1:**
	•	After the transcription process, LLaMA 3.1 will be used to refine the text by correcting any typographical errors and improving overall sentence structure and clarity. This ensures a more polished and accurate transcription output.
2.	**Sentiment Analysis Based on Audio Emotions:**
	•	In addition to transcribing speech, a sentiment analysis feature will be added to detect emotional tone (e.g., happiness, sadness, anger) from the audio using emotion-detection models. This will provide a richer context to the transcription by capturing the emotional state of each speaker.


## Usage

1.**Install the required packages:**
  ```bash
    pip install -r requirements.txt
```
2.**You will need to set up the Hugging Face transformers and authenticate with your Hugging Face account for access to the models. This can be done as follows:**
  ```bash
    huggingface-cli login
```
3.**Run the Streamlit app:**
```bash
streamlit run voice.py
```
## Using the App
1.	**Once the app is running, upload an audio file using the file uploader. Supported formats are .mp3, .wav, .flac, .m4a, and .ogg.**
2.	**The app will guide you through the following steps:**
	•	Convert to WAV: The uploaded audio is converted into WAV format.
	•	Audio Enhancement: The app will enhance the audio by reducing noise and normalizing the audio levels. Visual plots will show the original and enhanced audio signals. (still on dev phase)
	•	Speaker Diarization: Pyannote’s Speaker Diarization model will be applied to identify different speakers in the audio.
	•	Generate RTTM File: The app will generate an RTTM file marking the start and end times of each speaker’s segments.
	•	Audio Splitting: The app will split the audio into smaller chunks based on the speaker diarization and prepare them for transcription.
	•	Transcription: The Whisper model will transcribe the speech from each speaker segment into text.
3.	**After processing, you will receive:**
	•	A detailed transcription with speaker labels.
	•	The RTTM file, which marks the start and end times of each speaker segment.
	•	Enhanced audio files corresponding to each speaker. (still on dev phase)



