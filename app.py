import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import queue
import tempfile

st.title("Speech To Text Multilingual Indonesian and English")
st.text("Speech To Text Multilingual ini menggunakan model Whisper yang telah di Fine Tuned")

# Load model dan processor
@st.cache_resource
def load_model():
    model = WhisperForConditionalGeneration.from_pretrained("model_STT/")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    return model.eval(), processor

model, processor = load_model()

# Queue untuk menyimpan frame audio
audio_queue = queue.Queue()

class MicrophoneProcessor(AudioProcessorBase):
    def recv(self, frame):
        # frame: av.AudioFrame
        arr = frame.to_ndarray()
        audio_queue.put(arr)
        return frame

st.sidebar.title("Input Audio")
mode = st.sidebar.radio("Jenis input:", ["Rekam Mikrofon", "Upload File"])

if mode == "Rekam Mikrofon":
    st.sidebar.write("Klik **Start** untuk mulai rekam, lalu **Stop** setelah selesai.")
    webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=MicrophoneProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
    if st.sidebar.button("Transkripsi Recording"):
        # Ambil semua audio yang direkam
        arrs = []
        while not audio_queue.empty():
            arrs.append(audio_queue.get())
        if not arrs:
            st.warning("Tidak ada audio yang direkam!")
        else:
            waveform = np.concatenate(arrs, axis=1)
            sr = 48000  # default mic sampling rate
            if waveform.shape[0] > 1:
                waveform = waveform.mean(axis=0, keepdims=True)
            if sr != 16000:
                waveform = torch.from_numpy(waveform)
                waveform = Resample(sr, 16000)(waveform).numpy()
                sr = 16000
            st.audio(waveform, format="audio/wav", sample_rate=sr)
            segments = processor(
                waveform.squeeze(),
                sampling_rate=sr,
                return_tensors="pt"
            )
            input_features = segments.input_features.to(model.device)
            with torch.no_grad():
                ids = model.generate(input_features=input_features, task="transcribe")
            text = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            st.subheader("Hasil Transkripsi")
            st.write(text)

elif mode == "Upload File":
    audio_file = st.sidebar.file_uploader("Upload WAV/MP3/M4A", type=["wav", "mp3", "m4a"])
    if audio_file:
        st.audio(audio_file)
        if st.sidebar.button("Transkripsi File"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                fname = tmp.name
            waveform, sr = torchaudio.load(fname)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = Resample(sr, 16000)(waveform)
                sr = 16000
            segments = processor(
                waveform.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt"
            )
            input_features = segments.input_features.to(model.device)
            with torch.no_grad():
                ids = model.generate(input_features=input_features, task="transcribe")
            text = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            st.subheader("Hasil Transkripsi")
            st.write(text)

else:
    st.info("Silakan pilih input audio dari sidebar.")
