import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
# Sửa lỗi import chuẩn xác cho MoviePy
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH ---
st.set_page_config(page_title="DAT Media AI Workflow", layout="wide", page_icon="🛡️")

# --- HÀM HỖ TRỢ AN TOÀN ---
def safe_generate_content(model_name, prompt):
    """Hàm này tự động đổi sang model cũ nếu model mới bị lỗi 404"""
    try:
        model = genai.GenerativeModel(model_name)
        return model.generate_content(prompt)
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            st.warning(f"⚠️ Model {model_name} chưa sẵn sàng ở vùng này, đang chuyển sang 'gemini-pro'...")
            fallback_model = genai.GenerativeModel("gemini-pro")
            return fallback_model.generate_content(prompt)
        else:
            raise e

def get_image_url(keyword):
    """Lấy ảnh minh họa miễn phí"""
    clean_keyword = keyword.replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{clean_keyword}?width=1280&height=720&nologo=true"

def create_video_from_script(script_data):
    """Dựng video từ kịch bản"""
    clips = []
    try:
        lines = script_data.strip().split('\n')
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    img_prompt = parts[0].replace("Scene", "").replace(":", "").strip()
                    voice_text = parts[1].strip()
                    
                    # 1. Tạo Audio
                    tts = gTTS(text=voice_text, lang='vi')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                        tts.save(audio_file.name)
                        audio_path = audio_file.name
                    
                    # 2. Tải ảnh
                    img_url = get_image_url(img_prompt)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
                        img_file.write(requests.get(img_url).content)
                        img_path = img_file.name
                    
                    # 3. Ghép Clip
                    audio_clip = AudioFileClip(audio_path)
                    clip = ImageClip(img_path).set_duration(audio_clip.duration + 0.5)
                    clip = clip.set_audio(audio_clip)
                    clip = clip.set_fps(24)
                    clips.append(clip)
        
        if clips:
            final_video = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                final_video.write_videofile(temp_video.name, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')
                return temp_video.name
    except Exception as e:
        st.error(f"Lỗi dựng phim: {str(e)}")
        return None

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.success("✅ Đã kết nối API")
    else:
        key = st.text_input("Nhập API Key", type="password")
        if key: genai.configure(api_key=key)

st.title("🛡️ AI Content Generator: Bảo Hiểm")

# --- KHÔI PHỤC ĐẦY ĐỦ TÍNH NĂNG ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Nhập yêu cầu")
    keyword = st.text_input("Chủ đề", "Bảo hiểm thai sản")
    sector = st.selectbox("Lĩnh vực", ["Nhân thọ", "Phi nhân thọ", "Sức khỏe"])
    
    # Đã trả lại menu chọn đầy đủ
    content_type = st.radio("Loại nội dung", ["Clip (Video)", "Bài Website", "Bài Facebook"])
    
    tone = st.select_slider("Tone giọng", ["Hài hước", "Đời thường", "Chuyên nghiệp", "Cảm động"])
    
    if st.button("🚀 XỬ LÝ NGAY"):
        st.session_state.processing = True

# --- XỬ LÝ KẾT QUẢ ---
if st.session_state.get('processing'):
    with col2:
        st.subheader("2. Kết quả AI")
        with st.spinner("Đang suy nghĩ..."):
            
            # Tạo prompt thông minh
            base_prompt = f"Vai trò: Chuyên gia bảo hiểm {sector}. Chủ đề: {keyword}. Tone giọng: {tone}. "
            
            if content_type == "Clip (Video)":
                prompt = base_prompt + "Viết kịch bản video ngắn. BẮT BUỘC định dạng từng dòng: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'"
            elif content_type == "Bài Website":
                prompt = base_prompt + "Viết bài chuẩn SEO, dài 800 từ. Có thẻ H1, H2 và đề xuất chỗ chèn ảnh."
            else:
                prompt = base_prompt + "Viết caption Facebook thu hút, nhiều emoji."

            # Gọi AI với cơ chế an toàn (Tự chuyển model nếu lỗi)
            try:
                response = safe_generate_content("gemini-1.5-flash", prompt)
                st.session_state.result = response.text
                st.session_state.type = content_type
            except Exception as e:
                st.error(f"Lỗi kết nối: {e}")

    # Hiển thị
    if 'result' in st.session_state:
        if st.session_state.type == "Clip (Video)":
            tab1, tab2 = st.tabs(["🎬 Xem Video Demo", "📝 Đọc Kịch bản"])
            with tab2:
                st.text_area("Kịch bản", st.session_state.result, height=300)
            with tab1:
                if st.button("🎥 Bấm để Dựng Video (Mất khoảng 1 phút)"):
                    video_file = create_video_from_script(st.session_state.result)
                    if video_file:
                        st.video(video_file)
        else:
            st.markdown(st.session_state.result)
