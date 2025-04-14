# 🍠 Smart Online Video Editor App with AI Features

import streamlit as st
import os
import tempfile
import subprocess
import whisper
import cv2
from PIL import Image
import base64
import moviepy.editor as mp
from streamlit_drawable_canvas import st_canvas
import google.generativeai as genai
from compressor import compress_video

st.set_page_config(page_title="✨ Smart Video Editor", layout="wide")
st.title("🎞 Smart Online Video Editor")
st.markdown("Style, analyze, and understand your video – all in one place.")

# Gemini API setup
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Sidebar options
st.sidebar.title("🕠 Tools")
tool = st.sidebar.radio("Choose a feature:", (
    "Compress Video",
    "Generate Subtitles",
    "AI Video Overview (Gemini API)",
    "Frame-by-Frame Viewer",
    "Trim Video",
    "Crop Video",
    "Add Filter (Grayscale)",
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
            os.sync() if hasattr(os, "sync") else None
            model = whisper.load_model("base")
            with st.spinner("Transcribing using Whisper..."):
                result = model.transcribe(temp_video_path, language=language)
                subtitle_text = result["text"]
            st.success("✅ Subtitles generated")
            st.text_area("Subtitles:", subtitle_text, height=300)
            st.download_button("⬇️ Download Subtitles", subtitle_text, file_name=filename + "_subtitles.txt")
        except Exception as e:
            st.error(f"Subtitle generation failed: {e}")

    elif tool == "AI Video Overview (Gemini API)":
        st.subheader("🤖 AI Video Summary (Experimental)")
        st.info("This feature extracts subtitles and sends them to Gemini API for summarization.")

        try:
            with st.spinner("Transcribing with Whisper..."):
                model = whisper.load_model("base")
                result = model.transcribe(temp_video_path)
                transcript = result["text"]

            st.text_area("Transcript:", transcript, height=200)

            with st.spinner("Sending to Gemini..."):
                gemini_model = genai.GenerativeModel("gemini-pro")
                response = gemini_model.generate_content(f"Summarize this transcript:\n{transcript}")

            st.success("✅ Summary generated")
            st.text_area("Gemini Summary:", response.text, height=200)
        except Exception as e:
            st.error(f"AI Summary failed: {e}")

    elif tool == "Frame-by-Frame Viewer":
        st.subheader("🧱 Frame Viewer")
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
        st.info("Draw a rectangle on the first frame to select crop area")
        video = mp.VideoFileClip(temp_video_path)
        frame = video.get_frame(1)
        frame_img = Image.fromarray(frame)

        if frame_img.width > 1920:
            resize_ratio = 1920 / frame_img.width
            frame_img = frame_img.resize((1920, int(frame_img.height * resize_ratio)))

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            background_image=frame_img,
            update_streamlit=True,
            height=frame_img.height,
            width=frame_img.width,
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][0]
            x1 = int(obj["left"])
            y1 = int(obj["top"])
            x2 = int(x1 + obj["width"])
            y2 = int(y1 + obj["height"])

            st.write(f"Cropping rectangle: ({x1}, {y1}) to ({x2}, {y2})")
            cropped_path = temp_video_path.replace(".mp4", "_cropped.mp4")

            if st.button("Crop"):
                with st.spinner("Cropping video..."):
                    cropped = video.crop(x1=x1, y1=y1, x2=x2, y2=y2)
                    cropped.write_videofile(cropped_path, codec="libx264")
                    st.success("✅ Cropping complete")
                    st.video(cropped_path)

    elif tool == "Add Filter (Grayscale)":
        st.subheader("🎰 Add Grayscale Filter")
        st.info("Converting video to grayscale...")
        gray_path = temp_video_path.replace(".mp4", "_gray.mp4")
        if st.button("Apply Grayscale"):
            with st.spinner("Processing..."):
                clip = mp.VideoFileClip(temp_video_path).fx(mp.vfx.blackwhite)
                clip.write_videofile(gray_path, codec="libx264")
                st.success("✅ Grayscale filter applied")
                st.video(gray_path)

else:
    st.info("👈 Upload a video to get started")

st.markdown("---")
