import streamlit as st
from transformers import pipeline
from pytube import YouTube
from pydub import AudioSegment
from audio_extract import extract_audio
import google.generativeai as google_genai
import os
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
google_genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="VidText")


def youtube_video_downloader(url):
    yt_vid = YouTube(url)
    title = yt_vid.title
    vid_dld = (
        yt_vid.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )
    vid_dld = vid_dld.download()
    return vid_dld, title


def audio_extraction(video_file, output_format):
    # temp_filename = video_file.name
    # video_path = f"{temp_filename}"
    audio = extract_audio(
        input_path=os.fspath(video_file),
        output_path=f"{str(video_file)[:-4]}.mp3",
        output_format=f"{output_format}",
    )
    
    return audio


def audio_processing(mp3_audio):
    audio = AudioSegment.from_file(mp3_audio, format="mp3")
    wav_file = "audio_file.wav"
    audio = audio.export(wav_file, format="wav")
    return wav_file


@st.cache_resource
def load_asr_model():
    asr_model = pipeline(
        task="automatic-speech-recognition", model="openai/whisper-large-v3"
    )
    
    return asr_model


def transcribe_video(processed_audio):
    transcriber_model = load_asr_model()
    text_extract = transcriber_model(processed_audio)
    return text_extract["text"]


def generate_ai_summary(transcript):
    model = google_genai.GenerativeModel("gemini-pro")
    model_response = model.generate_content(
        [f"Give a summary of the text {transcript}"], stream=True
    )
    return model_response.text


# Streamlit UI

youtube_url_tab, file_select_tab, audio_file_tab = st.tabs(
    ["Youtube url", "Video file", "Audio file"]
)

with youtube_url_tab:
    url = st.text_input("Enter the Youtube url")

    try:
        yt_video, title = youtube_video_downloader(url)
        if url:
            if st.button("Transcribe", key="yturl"):
                with st.spinner("Transcribing..."):
                    audio = audio_extraction(yt_video, "mp3")
                    audio = audio_processing(audio)
                    ytvideo_transcript = transcribe_video(audio)
                st.success(f"Transcription successful")
                st.write(ytvideo_transcript)
                if st.button("Generate Summary"):
                    summary = generate_ai_summary(ytvideo_transcript)
                    st.write(summary)

    except Exception as e:
        st.error(f"Enter a valid url: {e}")

# Video file transcription

with file_select_tab:
    video_file = st.file_uploader("Upload video file", type="mp4")
    try:
        # video_file = video_file.getvalue()
        with open(video_file, "wb") as f:
            f.write(video_file.getvalue())

        if video_file:
            if st.button("Transcribe", key="vidfile"):
                with st.spinner("Transcribing..."):
                    audio = audio_extraction(video_file, "mp3")
                    audio = audio_processing(audio)
                    video_transcript = transcribe_video(audio)
                    st.success(f"Transcription successful")
                    st.write(video_transcript)
                    if st.button("Generate Summary", key="ti2"):
                        summary = generate_ai_summary(video_transcript)
                        st.write(summary)

    except Exception as e:
        st.error(e)

# Audio transcription
with audio_file_tab:
    audio_file = st.file_uploader("Upload audio file", type="mp3")

    if audio_file:
        if st.button("Transcribe", key="audiofile"):
            with st.spinner("Transcribing..."):
                processed_audio = audio_processing(audio_file)
                audio_transcript = transcribe_video(processed_audio)
                st.success(f"Transcription successful")
                st.write(audio_transcript)

                if st.button("Generate Summary", key="ti1"):
                    summary = generate_ai_summary(audio_transcript)
                    st.write(summary)
