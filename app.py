import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
import re # Thư viện xử lý văn bản để tìm chỗ chèn ảnh
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH ---
st.set_page_config(page_title="DAT Media AI Workflow", layout="wide", page_icon="🛡️")

# --- CSS LÀM ĐẸP ---
st.markdown("""
    <style>
    .stButton>button {background-color: #FF4B4B; color: white; font-weight: bold; border-radius: 8px;}
    img {border-radius: 10px; margin-top: 10px; margin-bottom: 10px;}
    .caption {font-style: italic; color: #666; font-size: 0.9em; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.title("⚙️ Cấu hình hệ thống")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Đã kết nối API từ hệ thống")
    else:
        api_key = st.text_input("Nhập API Key", type="password")
    
    st.divider()
    # Tự động quét và chọn model
    available_models = ["models/gemini-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            all_models = genai.list_models()
            available_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        except: pass
    
    selected_model = st.selectbox("Mô hình xử lý:", available_models, index=0)

# --- HÀM XỬ LÝ ẢNH & VIDEO ---

def get_image_url(prompt, width=1280, height=720):
    """Tạo URL ảnh từ Pollinations"""
    clean_prompt = prompt.replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&nologo=true"

def render_mixed_content(text):
    """
    Hàm thông minh: Đọc văn bản, tìm thẻ {{IMAGE: ...}} để hiển thị ảnh thực tế
    """
    # Tách văn bản thành các đoạn dựa trên thẻ {{IMAGE: ...}}
    parts = re.split(r'\{\{IMAGE: (.*?)\}\}', text)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Đây là phần văn bản thường
            if part.strip():
                st.markdown(part)
        else:
            # Đây là phần mô tả ảnh (nằm trong thẻ)
            img_prompt = part.strip()
            # Hiển thị ảnh minh họa (Size 800x450 cho bài viết)
            img_url = get_image_url(img_prompt, width=800, height=450)
            st.image(img_url, caption=f"Minh họa do AI tạo: {img_prompt}", use_container_width=True)

def create_video_from_script(script_data):
    """Dựng video từ kịch bản"""
    clips = []
    try:
        lines = script_data.strip().split('\n')
        for line in lines:
            if "|" in line and "Scene" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    img_prompt = parts[0].replace("Scene", "").replace(":", "").strip()
                    voice_text = parts[1].strip()
                    
                    tts = gTTS(text=voice_text, lang='vi')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                        tts.save(audio_file.name)
                        audio_path = audio_file.name
                    
                    img_url = get_image_url(img_prompt)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
                        img_file.write(requests.get(img_url).content)
                        img_path = img_file.name
                    
                    audio_clip = AudioFileClip(audio_path)
                    clip = ImageClip(img_path).set_duration(audio_clip.duration + 0.5).set_audio(audio_clip).set_fps(24)
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
st.title("🛡️ AI Content Generator: Bảo Hiểm & Tài Chính")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Đầu vào nội dung")
    keyword = st.text_input("Từ khóa / Chủ đề", "Bảo hiểm sức khỏe cho gia đình")
    sector = st.selectbox("Lĩnh vực", ["Bảo hiểm Nhân thọ", "Bảo hiểm Phi nhân thọ", "Chăm sóc sức khỏe"])
    
    content_type = st.radio("Định dạng", ["Clip (Video)", "Bài Website", "Bài Facebook"])
    
    tone_key = st.selectbox("Tone giọng", ["Chuyên nghiệp, Tin cậy", "Đời thường, Gần gũi", "Kể chuyện cảm động"])
    
    # Tùy biến Prompt nâng cao để ép AI sinh ra thẻ ảnh
    extra_prompt = ""
    if content_type == "Clip (Video)":
        duration = st.slider("Thời lượng (s)", 30, 90, 45)
        extra_prompt = f"Viết kịch bản Video {duration}s. Cấu trúc mỗi dòng: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'"
    elif content_type == "Bài Website":
        words = st.number_input("Số từ", 500, 2000, 800)
        # Prompt quan trọng: Dạy AI cách đánh dấu chỗ chèn ảnh
        extra_prompt = f"""
        Viết bài chuẩn SEO {words} từ. 
        YÊU CẦU HÌNH ẢNH:
        1. Bài viết phải có ít nhất 2 hình ảnh minh họa xen kẽ trong nội dung.
        2. Tại vị trí muốn chèn ảnh, hãy viết CHÍNH XÁC cú pháp sau: {{IMAGE: mô tả hình ảnh chi tiết bằng tiếng Anh}}.
        3. Ngay dòng dưới thẻ ảnh, hãy viết chú thích (Caption) bắt đầu bằng 'Chú thích:'.
        """
    else:
        extra_prompt = "Viết bài Facebook kèm 1 ảnh vuông (Mô tả ảnh ở cuối bài). Dùng nhiều emoji."

    btn_process = st.button("🚀 XỬ LÝ NGAY")

# --- XỬ LÝ KẾT QUẢ ---
with col2:
    st.subheader("2. Kết quả")
    
    if btn_process:
        if not api_key:
            st.error("Thiếu API Key")
        else:
            with st.spinner("AI đang sáng tạo nội dung và vẽ ảnh..."):
                try:
                    model = genai.GenerativeModel(selected_model)
                    full_prompt = f"Vai trò: Chuyên gia {sector}. Chủ đề: {keyword}. Tone: {tone_key}. {extra_prompt}"
                    
                    response = model.generate_content(full_prompt)
                    st.session_state.result_text = response.text
                    st.session_state.content_type = content_type
                    st.session_state.keyword = keyword # Lưu từ khóa để vẽ ảnh featured
                    st.success("Hoàn thành!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    if 'result_text' in st.session_state:
        # 1. XỬ LÝ VIDEO
        if st.session_state.content_type == "Clip (Video)":
            tab1, tab2 = st.tabs(["🎬 Video Demo", "📝 Kịch bản"])
            with tab1:
                if st.button("🎥 Dựng Video ngay"):
                    with st.spinner("Đang xử lý..."):
                        v_path = create_video_from_script(st.session_state.result_text)
                        if v_path: st.video(v_path)
            with tab2:
                st.text_area("Source", st.session_state.result_text, height=400)
        
        # 2. XỬ LÝ BÀI WEBSITE (CÓ ẢNH THỰC TẾ)
        elif st.session_state.content_type == "Bài Website":
            # Hiển thị Ảnh Featured đầu tiên (Cố định 1200x628)
            st.markdown("### 🖼️ Ảnh Featured (Ảnh bìa)")
            featured_url = get_image_url(st.session_state.keyword + " insurance professional high quality", width=1200, height=628)
            st.image(featured_url, caption="Ảnh đại diện bài viết (1200x628)", use_container_width=True)
            
            st.divider()
            st.markdown("### 📄 Nội dung chi tiết")
            # Gọi hàm thông minh để hiển thị bài viết kèm ảnh minh họa
            render_mixed_content(st.session_state.result_text)
            
        # 3. XỬ LÝ FACEBOOK
        else:
            st.info("Ảnh vuông cho Facebook:")
            fb_url = get_image_url(st.session_state.keyword + " insurance flat lay aesthetic", width=1080, height=1080)
            st.image(fb_url, width=400, caption="Ảnh vuông 1:1")
            st.markdown(st.session_state.result_text)
