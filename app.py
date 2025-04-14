import streamlit as st
import os
import tempfile
import subprocess
import whisper
import cv2
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import base64
import moviepy.editor as mp
import google.generativeai as genai
from compressor import compress_video
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

st.set_page_config(page_title="✨ Smart Video Editor", layout="wide")
st.title("🎮 Smart Online Video Editor")
st.markdown("Style, analyze, and understand your video – all in one place.")

# Gemini API setup
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Sidebar options
st.sidebar.title("🕠 Tools")
tool = st.sidebar.radio("Choose a feature:", (
    "Compress Video",
    "Generate Subtitles",
    "Video Overview",
    "Frame-by-Frame Viewer",
    "Trim Video",
    "Crop Video",
    "Add Filter",
    "AI Stylization",
))

uploaded_file = st.file_uploader("📄 Upload your video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    filename = uploaded_file.name.rsplit(".", 1)[0]

    if tool == "Compress Video":
        st.subheader("📦 Compressing your video")
        with st.spinner("Compressing using compressor.py logic..."):
            compressed_path = temp_video_path.replace(".mp4", "_compressed.mp4")
            try:
                compress_video(temp_video_path, compressed_path)
                if os.path.exists(compressed_path):
                    st.success("✅ Compression complete")
                    st.video(compressed_path)
                    with open(compressed_path, "rb") as f:
                        st.download_button("⬇️ Download Compressed Video", f, file_name=filename + "_compressed.mp4")
                else:
                    st.error("Compression failed: Output file not created.")
            except Exception as e:
                st.error(f"Compression failed: {str(e)}")

    elif tool == "Generate Subtitles":
        st.subheader("📝 Subtitle Generator")
        language = st.selectbox("Select language (for better accuracy):", ["en", "hi", "es", "fr", "de", "zh"])

        try:
            model = whisper.load_model("base")
            with st.spinner("Transcribing using Whisper..."):
                result = model.transcribe(temp_video_path, language=language)
                subtitle_text = result["text"]
            st.success("✅ Subtitles generated")
            st.text_area("Subtitles:", subtitle_text, height=300)
            st.download_button("⬇️ Download Subtitles", subtitle_text, file_name=filename + "_subtitles.txt")
        except Exception as e:
            st.error(f"Subtitle generation failed: {e}")

    elif tool == "Video Overview":
        st.subheader("🧐 Video Summary")
        st.info("This feature extracts subtitles and sends them to Gemini API for summarization.")

        try:
            with st.spinner("Transcribing with Whisper..."):
                model = whisper.load_model("base")
                result = model.transcribe(temp_video_path)
                transcript = result["text"]

            st.text_area("Transcript:", transcript, height=200)

            with st.spinner("Sending to Gemini..."):
                gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                response = gemini_model.generate_content(f"Summarize this transcript:\n{transcript}")

            st.success("✅ Summary generated")
            st.text_area("Gemini Summary:", response.text, height=200)
        except Exception as e:
            st.error(f"AI Summary failed: {e}")

    elif tool == "Frame-by-Frame Viewer":
        st.subheader("🧡 Frame Viewer")
        cap = cv2.VideoCapture(temp_video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = st.slider("Select frame index", 0, total - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption=f"Frame {idx}", use_column_width=True)
        else:
            st.error("Failed to extract frame.")

    elif tool == "Trim Video":
        st.subheader("✂️ Trim Video")
        video = mp.VideoFileClip(temp_video_path)
        st.video(temp_video_path)
        st.markdown("Select start and end time to trim the video")
        start_time = st.slider("Start time (seconds):", 0, int(video.duration) - 1, 0)
        end_time = st.slider("End time (seconds):", start_time + 1, int(video.duration), int(video.duration))
        trimmed_path = temp_video_path.replace(".mp4", "_trimmed.mp4")

        if st.button("Trim"):
            with st.spinner("Trimming video..."):
                trimmed = video.subclip(start_time, end_time)
                trimmed.write_videofile(trimmed_path, codec="libx264")
                st.success("✅ Trim complete")
                st.video(trimmed_path)

    elif tool == "Crop Video":
        st.subheader("🖼️ Crop Video")
        st.video(temp_video_path)
        cap = cv2.VideoCapture(temp_video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            st.image(image, caption="First Frame", use_column_width=True)
            st.markdown("Specify crop coordinates:")
            x1 = st.number_input("x1", value=0, min_value=0)
            y1 = st.number_input("y1", value=0, min_value=0)
            x2 = st.number_input("x2", value=image.width, min_value=1)
            y2 = st.number_input("y2", value=image.height, min_value=1)

            if st.button("Crop"):
                with st.spinner("Cropping video..."):
                    video = mp.VideoFileClip(temp_video_path)
                    cropped = video.crop(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
                    cropped_path = temp_video_path.replace(".mp4", "_cropped.mp4")
                    cropped.write_videofile(cropped_path, codec="libx264")
                    st.success("✅ Cropping complete")
                    st.video(cropped_path)

    elif tool == "Add Filter":
        st.subheader("🎰 Add Video Filter")
        filter_choice = st.selectbox("Choose filter:", ["Grayscale", "Sepia", "Invert", "Brighten"])
        filtered_path = temp_video_path.replace(".mp4", f"_{filter_choice.lower()}.mp4")

        if st.button("Apply Filter"):
            clip = mp.VideoFileClip(temp_video_path)
            if filter_choice == "Grayscale":
                clip = clip.fx(mp.vfx.blackwhite)
            elif filter_choice == "Invert":
                clip = clip.fl_image(lambda f: 255 - f)
            elif filter_choice == "Sepia":
                def sepia(img):
                    img = np.array(img)
                    sepia_filter = np.array([[0.393, 0.769, 0.189],
                                             [0.349, 0.686, 0.168],
                                             [0.272, 0.534, 0.131]])
                    return np.clip(img.dot(sepia_filter.T), 0, 255).astype(np.uint8)
                clip = clip.fl_image(sepia)
            elif filter_choice == "Brighten":
                clip = clip.fl_image(lambda f: np.clip(f * 1.2, 0, 255))

            clip.write_videofile(filtered_path, codec="libx264")
            st.success("✅ Filter applied")
            st.video(filtered_path)

    elif tool == "Ask Questions About Video":
        st.subheader("💬 Ask Questions About Video")

        # Step 1: Extract subtitles
        st.info("Extracting subtitles... this might take a moment.")
        try:
            model = whisper.load_model("base")
            with st.spinner("Transcribing video with Whisper..."):
                result = model.transcribe(temp_video_path)
                transcript = result["text"]

            st.text_area("Transcript:", transcript, height=200)

            # Step 2: Submit subtitles to Gemini API for processing
            st.info("Sending transcript to Gemini API for analysis...")
            try:
                gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                gemini_response = gemini_model.generate_content(f"Analyze the following video transcript and answer questions about it:\n{transcript}")
                st.success("✅ Transcript analysis complete.")
                st.text_area("Gemini Analysis:", gemini_response.text, height=200)

                # Step 3: Chatbot interface for questions
                st.info("Ask a question about the video:")
                user_question = st.text_input("Your question:")

                if user_question:
                    try:
                        response = gemini_model.generate_content(f"Answer the following question based on the video transcript: {user_question}\nTranscript:\n{transcript}")
                        st.success("✅ Answer generated.")
                        st.text_area("AI's Answer:", response.text, height=150)
                    except Exception as e:
                        st.error(f"Error while generating the answer: {e}")

            except Exception as e:
                st.error(f"Error while analyzing the transcript with Gemini: {e}")
        except Exception as e:
            st.error(f"Error while extracting subtitles from the video: {e}")
else:
    st.info("👈 Upload a video to get started")

st.markdown("---")
