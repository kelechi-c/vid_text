import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq
from pytube import YouTube
from pydub import AudioSegment
from audio_extract import extract_audio
from tqdm import tqdm
import os

st.set_page_config(
    page_title="VidText",   
)

class VideoTranscriber:
    def __init__(self):
        self.url = None
        self.title = None
        self.processed_audio = None
        self.text_extract = None

    def youtube_video_downloader(self, url):
        yt_vid = YouTube(url=url)
        self.title = yt_vid.title
        vid_dld = yt_vid.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        vid_dld = vid_dld.download()
        return vid_dld, self.title

    def audio_extraction(self, vid_dld):
        audio = extract_audio(input_path=vid_dld, output_path=f'{self.title}1.mp3')
        audio = AudioSegment.from_file(audio, format='mp3')
        wav_file = f'{self.title}.wav'
        self.processed_audio = audio.export(wav_file, format="wav")
        return self.processed_audio

    def transcriber(self, processed_audio):
        transcriber_model = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
        self.text_extract = transcriber_model(processed_audio)
        return self.text_extract['text']

    def summarizer(self):
        summarizer_model = pipeline(
            task="summarization",
            model="google-t5/t5-base",
            tokenizer="google-t5/t5-base",
        )

        summary = summarizer_model(self.text_extract)
        return summary['summary_text']


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

def audio_extraction(vid_dld, title):
    audio = extract_audio(input_path=vid_dld, output_path=f'{title}1.mp3')
    audio = AudioSegment.from_file(audio, format='mp3')
    wav_file = f'{title}.wav'
    processed_audio = audio.export(wav_file, format="wav")
    return processed_audio


@st.cache_resource
def transcriber_pass(processed_audio):
    transcriber_model = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
    text_extract = transcriber_model(processed_audio)
    return text_extract['text']

# video_transcriber = VideoTranscriber()

@st.cache_resource
def transcribe_video(video_file):
    processed_audio = audio_extraction(video_file)
    text = transcriber_pass(processed_audio)
    return text


# Streamlit UI

url_input_tab, file_select_tab = st.tabs(["Youtube url", "Video file"])

# with url_input_tab:
#     url = st.text_input("Enter the Youtube url")
#     yt_video, title = youtube_video_downloader(url)
#     if yt_video:
#         if st.button("Transcribe"):
#             with st.spinner("Transcribing..."):
#                 ytvideo_transcript = transcribe(yt_video)
#             st.success(f"Transcription successful")
#             st.write(ytvideo_transcript)


# Audio only transcription
with file_select_tab:
    video_file = st.file_uploader("Upload video file", type="mp4")
    
    if video_file:
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                video_transcript = transcribe_video(video_file) 
                st.success(f"Transcription successful")
                st.write(video_transcript)
