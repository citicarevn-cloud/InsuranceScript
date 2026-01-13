import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
# --- SỬA LỖI QUAN TRỌNG TẠI ĐÂY ---
# Đã xóa ConcatenateAudioClip và thay bằng concatenate_videoclips chuẩn
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH ---
st.set_page_config(page_title="AI Bảo Hiểm - Video Generator", layout="wide", page_icon="🎬")

# --- HÀM HỖ TRỢ: TÌM ẢNH MIỄN PHÍ ---
def get_image_url(keyword):
    clean_keyword = keyword.replace(" ", "%20")
    # Dùng Pollinations AI để vẽ ảnh (Miễn phí, không cần key)
    return f"https://image.pollinations.ai/prompt/{clean_keyword}?width=1280&height=720&nologo=true"

# --- HÀM HỖ TRỢ: TẠO VIDEO ---
def create_video_segment(text, image_prompt):
    # 1. Tạo Audio từ Text
    tts = gTTS(text=text, lang='vi')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
        tts.save(audio_file.name)
        audio_path = audio_file.name

    # 2. Tải ảnh về
    img_url = get_image_url(image_prompt)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
        img_data = requests.get(img_url).content
        img_file.write(img_data)
        img_path = img_file.name

    # 3. Dựng Clip bằng MoviePy
    audio_clip = AudioFileClip(audio_path)
    # Ảnh hiện lâu bằng độ dài audio + 0.5s nghỉ
    clip_duration = audio_clip.duration + 0.5
    
    video_clip = ImageClip(img_path).set_duration(clip_duration)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip = video_clip.set_fps(24)
    
    return video_clip

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    # Tự động lấy Key từ Secrets
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.success("✅ Đã kết nối API")
    else:
        api = st.text_input("Nhập API Key thủ công")
        if api: genai.configure(api_key=api)

st.title("🎬 AI Tạo Video Demo Bảo Hiểm")
st.caption("Dành cho DAT Media: Tạo bản nháp video nhanh chóng từ kịch bản.")

col1, col2 = st.columns([1, 1.5])

with col1:
    topic = st.text_input("Chủ đề video", "Bảo hiểm thai sản trọn gói")
    
    if st.button("🎥 LÊN KỊCH BẢN & DỰNG VIDEO"):
        st.session_state.processing = True

# --- XỬ LÝ LOGIC ---
if st.session_state.get('processing'):
    with col2:
        # BƯỚC 1: VIẾT KỊCH BẢN
        with st.status("1. AI đang viết kịch bản...", expanded=True) as status:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Viết kịch bản video ngắn về: {topic}.
                Chia làm đúng 3 phân cảnh (Scene).
                Trả về định dạng thuần (Bắt buộc):
                Scene 1: [Mô tả hình ảnh tiếng Anh ngắn gọn để AI vẽ] | [Lời bình tiếng Việt]
                Scene 2: [Mô tả hình ảnh tiếng Anh ngắn gọn để AI vẽ] | [Lời bình tiếng Việt]
                Scene 3: [Mô tả hình ảnh tiếng Anh ngắn gọn để AI vẽ] | [Lời bình tiếng Việt]
                Không thêm lời chào hay ký tự thừa.
                """
                response = model.generate_content(prompt)
                script_content = response.text
                st.code(script_content, language="text")
                status.update(label="✅ Đã xong kịch bản!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Lỗi kịch bản: {e}")
                st.stop()

        # BƯỚC 2: DỰNG VIDEO
        with st.status("2. Đang vẽ ảnh & Dựng video (Khoảng 1-2 phút)...", expanded=True) as status:
            try:
                lines = script_content.strip().split('\n')
                clips = []
                
                for line in lines:
                    if "|" in line:
                        parts = line.split("|")
                        img_prompt = parts[0].replace("Scene", "").replace(":", "").strip()
                        voice_text = parts[1].strip()
                        
                        st.write(f"🎨 Đang vẽ: {img_prompt}")
                        clip = create_video_segment(voice_text, img_prompt)
                        clips.append(clip)
                
                if clips:
                    # Nối các đoạn lại thành 1 video
                    final_video = concatenate_videoclips(clips, method="compose")
                    
                    # Xuất file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                        final_video.write_videofile(temp_video.name, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')
                        st.session_state.video_path = temp_video.name
                    
                    status.update(label="✅ Đã dựng xong Video!", state="complete")
                else:
                    st.warning("AI trả về kịch bản không đúng định dạng. Hãy thử lại.")
                    
            except Exception as e:
                st.error(f"Lỗi dựng phim: {str(e)}")

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state.get('video_path'):
    with col2:
        st.success("🎉 XONG! VIDEO CỦA BẠN ĐÂY:")
        st.video(st.session_state.video_path)
        
        with open(st.session_state.video_path, "rb") as file:
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=file,
                file_name="demo_baohiem.mp4",
                mime="video/mp4"
            )
