## vidkit-v2 (formerly VidText)

### Description
I am not usually chanced enough to listen to/watch long video/audio files :).
So I built this, an open-source AI project for transcribing video and audio files, even youtube videos (from the link). 
It utilizes a pretrained ASR(Automatic Speech Recognition) model from the Huggingface transformers library, Streamlit for the web UI, and other python libraries for audio processing and model loading.

**[27-10-24]** ->  I rewrote the entire thing, with extra features,
 and a faster model (Moonshine-base with a JAX backend)

**NB**: Larger files and longer videos take a longer time to transcribe...(of course)

Check it out => **[**vidkit**](https://huggingface.co/spaces/tensorkelechi/vidkitv2)** on **HuggingFace spaces**