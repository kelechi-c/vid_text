import moonshine
import streamlit as st
import time, os
from pytube import YouTube
from pydub import AudioSegment


os.environ["KERAS_BACKEND"] = "jax"

st.set_page_config(page_title="vidkit_v2")

st.title("vidkit")
st.write("App for video/audio transcription(Youtube, mp4, mp3). Uses the Moonshine ASR model")
st.write("[built to solve a personal problem, might be useful to others too :)]")


def youtube_video_downloader(url: str):
    
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


def audio_extraction(video_file: str):
    audio = AudioSegment.from_file(video_file, format="mp4")
    audio_path = "audio.wav"
    audio.export(audio_path, format="wav")

    return audio_path


def audio_processing(mp3_audio):
    audio = AudioSegment.from_file(mp3_audio, format="mp3")
    wav_file = "audio_file.wav"
    audio = audio.export(wav_file, format="wav")
    return wav_file


# @st.cache_resource
# def load_asr_model():
#     stime = time.time()

#     asr_model = pipeline(
#         task="automatic-speech-recognition", model=""
#     )

#     time_taken = time.time() - stime
#     st.success(f"ASR model loaded in {time_taken}s")

#     return asr_model

# transcriber_model = load_asr_model()


def transcriber_pass(processed_audio):
    stime = time.time()
    
    # transcribe with moonshine
    text_extract = moonshine.transcribe(processed_audio, "moonshine/tiny")
    
    time_taken = time.time() - stime
    st.write(f'transcribed in {time_taken}s')
    
    return text_extract[0]


# Streamlit UI
youtube_url_tab, file_select_tab, audio_file_tab = st.tabs(
    ["Youtube URL", "Video file", "Audio file"]
)

with youtube_url_tab:
    url = st.text_input("Enter the Youtube url")

    try:
        yt_video, title = youtube_video_downloader(url)
        if url:
            if st.button("Transcribe", key="yturl"):
                with st.spinner("Transcribing..."):
                    with st.spinner("Extracting audio..."):
                        audio = audio_extraction(yt_video)
                    ytvideo_transcript = transcriber_pass(audio)

                st.write(f"Video title: {title}")
                st.write("___")
                
                st.markdown(
                    f"""
                          <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
                             <p> -> {ytvideo_transcript}</p>
                            </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(e)


# Video file transcription

with file_select_tab:
    uploaded_video_file = st.file_uploader("Upload video file", type="mp4")

    try:
        if uploaded_video_file:
            if st.button("Transcribe", key="vidfile"):
                with st.spinner("Transcribing..."):
                    with st.spinner("Extracting audio..."):
                        audio = audio_extraction(uploaded_video_file)

                    video_transcript = transcriber_pass(audio)
                    st.success(f"Transcription successful")
                    st.markdown(
                        f"""
                          <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
                             <p> -> {video_transcript}</p>
                            </div>
                            """,
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(e)


# Audio file transcription
with audio_file_tab:
    audio_file = st.file_uploader("Upload audio file", type="mp3")

    try:
        # ensure audio file is present
        if audio_file:
            if st.button("Transcribe", key="audiofile"):
                with st.spinner("Transcribing..."):
                    processed_audio = audio_processing(audio_file) # extract audio/preprocess
                    audio_transcript = transcriber_pass(processed_audio)
                    st.success(f"Transcription successful")
                    st.markdown(
                        f"""
                          <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
                             <p> -> {audio_transcript}</p>
                            </div>
                            """,
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(e)


# Footer
st.write("")
st.write("")
st.write("")

st.markdown(
    """
    <div style="text-align: center; padding: 1rem;">
        Project by <a href="https://github.com/kelechi-c" target="_blank" style="color: white; font-weight: bold; text-decoration: none;">
         tensor_kelechi</a>
    </div>
""",
    unsafe_allow_html=True,
)

# Arigato :)
