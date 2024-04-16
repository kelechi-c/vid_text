import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq
from pytube import YouTube
from pydub import AudioSegment
from audio_extract import extract_audio
from tqdm import tqdm
import os


st.set_page_config(
    page_title="VidText"
)

def youtube_video_downloader(url):
    yt_vid = YouTube(url)
    title = yt_vid.title
    vid_dld = (
        yt_vid.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()    
    )
    # vid_dld = vid_dld.download()
    return vid_dld, title


def audio_extraction(video_file, output_format):
    # temp_filename = video_file.name
    # video_path = f"{temp_filename}"
    audio = extract_audio(
        input_path=video_file, output_path=f"{video_file[:-4]}.mp3", output_format=f"{output_format}"
    )
    return audio


def audio_processing(mp3_audio):
    audio = AudioSegment.from_file(mp3_audio, format="mp3")
    wav_file = "audio_file.wav"
    audio = audio.export(wav_file, format="wav")
    return wav_file


@st.cache_resource
def transcribe_video(processed_audio):
    transcriber_model = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
    text_extract = transcriber_model(processed_audio)
    return text_extract['text']


# Streamlit UI

url_input_tab, file_select_tab, audio_file_tab = st.tabs(["Youtube url", "Video file", "Audio file"])

# with url_input_tab:video_path
#     url = st.text_input("Enter the Youtube url")
#     yt_video, title = youtube_video_downloader(url)
#     if yt_video:
#         if st.button("Transcribe"):
#             with st.spinner("Transcribing..."):
#                 ytvideo_transcript = transcribe(yt_video)
#             st.success(f"Transcription successful")
#             st.write(ytvideo_transcript)


# Video file transcription
with file_select_tab:
    video_file = st.file_uploader("Upload video file", type="mp4")
    
    
    if video_file:
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                audio = audio_extraction(video_file, "mp3")
                video_transcript = transcribe_video(audio) 
                st.success(f"Transcription successful")
                st.write(video_transcript)


# Audio transcription
with audio_file_tab:
    audio_file = st.file_uploader("Upload audio file", type="mp3")  

    if audio_file:
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                processed_audio = audio_processing(audio_file)
                audio_transcript = transcribe_video(processed_audio)
                st.success(f"Transcription successful")
                st.write(audio_transcript)


