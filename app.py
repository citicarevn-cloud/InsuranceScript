import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
import re
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media AI Workflow", layout="wide", page_icon="🛡️")

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%;}
    /* Style cho ảnh đẹp hơn */
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    .caption {text-align: center; color: #666; font-style: italic; font-size: 0.9em;}
    h1, h2, h3 {color: #333;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Đã kết nối API")
    else:
        api_key = st.text_input("Nhập API Key", type="password")

    # Tự động quét Model
    available_models = ["models/gemini-1.5-flash", "models/gemini-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            # Lấy danh sách model thực tế
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except: pass
        
    selected_model = st.selectbox("Chọn Model xử lý:", available_models, index=0)

# --- HÀM XỬ LÝ ẢNH & VIDEO (ĐÃ NÂNG CẤP) ---

def get_image_url(prompt, width=1280, height=720):
    """Tạo URL ảnh Pollinations với bộ lọc style"""
    # Thêm từ khóa style để ảnh đẹp, tránh người thật
    style = ", high quality illustration, isometric style, flat design, vector art, cinematic lighting"
    clean_prompt = (prompt + style).replace(" ", "%20")
    # Thêm seed ngẫu nhiên để ảnh không bị trùng
    seed = os.urandom(4).hex()
    return f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&nologo=true&seed={seed}"

def render_mixed_content(text):
    """
    Hàm hiển thị thông minh: Chấp nhận cả {IMAGE} và {{IMAGE}}
    """
    # Regex linh hoạt: Bắt ngoặc đơn { hoặc kép {{, theo sau là IMAGE:
    # (?s) cho phép dấu chấm khớp với dòng mới
    pattern = r'\{{1,2}IMAGE:?\s*(.*?)\}{1,2}'
    
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Phần văn bản
            if part.strip(): 
                st.markdown(part)
        else:
            # Phần mô tả ảnh
            img_prompt = part.strip()
            # Loại bỏ các ký tự thừa nếu có
            img_prompt = img_prompt.replace("}", "").replace("{", "")
            
            with st.container():
                st.write("") # Tạo khoảng trống
                # Vẽ ảnh ngay lập tức
                img_url = get_image_url(img_prompt, width=800, height=450)
                st.image(img_url, caption=f"🎨 Minh họa AI: {img_prompt[:50]}...", use_container_width=True)
                st.write("")

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
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        tts.save(f.name); audio_path = f.name
                    
                    img_url = get_image_url(img_prompt)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                        f.write(requests.get(img_url).content); img_path = f.name
                    
                    ac = AudioFileClip(audio_path)
                    clip = ImageClip(img_path).set_duration(ac.duration+0.5).set_audio(ac).set_fps(24)
                    clips.append(clip)
        
        if clips:
            final = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                final.write_videofile(f.name, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')
                return f.name
    except Exception as e:
        st.error(f"Lỗi video: {e}"); return None

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ AI Content Generator: Bảo Hiểm & Tài Chính")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Nhập yêu cầu")
    keyword = st.text_input("Chủ đề", "Bảo hiểm nhân thọ trọn đời")
    sector = st.selectbox("Lĩnh vực", ["Nhân thọ", "Phi nhân thọ", "Sức khỏe", "Tài chính"])
    
    content_type = st.radio("Định dạng", ["Bài Website chuẩn SEO", "Bài Facebook Viral", "Clip (Video)"])
    
    tone = st.select_slider("Tone giọng", ["Hài hước", "Đời thường", "Chuyên nghiệp", "Cảm động"])
    
    # Prompt mạnh mẽ
    extra_prompt = ""
    if content_type == "Clip (Video)":
        duration = st.slider("Giây", 30, 90, 45)
        extra_prompt = f"Viết kịch bản Video {duration}s. Cấu trúc mỗi dòng: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'"
    elif content_type == "Bài Website chuẩn SEO":
        words = st.number_input("Số từ", 500, 2000, 800)
        extra_prompt = f"""
        Viết bài chuẩn SEO {words} từ. 
        BẮT BUỘC CHÈN ẢNH MINH HỌA:
        Dùng thẻ {{IMAGE: mô tả ảnh tiếng Anh}} để chèn ít nhất 2 ảnh vào bài.
        Ví dụ: {{IMAGE: family protection umbrella illustration}}
        """
    else:
        extra_prompt = "Viết caption Facebook thu hút. Đề xuất 1 ảnh vuông cuối bài."

    btn_run = st.button("🚀 BẮT ĐẦU XỬ LÝ")

# --- KẾT QUẢ ---
with col2:
    st.subheader("2. Kết quả hiển thị")
    
    if btn_run:
        if not api_key: st.error("Chưa có API Key!")
        else:
            with st.spinner("Đang viết bài và vẽ ảnh..."):
                try:
                    model = genai.GenerativeModel(selected_model)
                    prompt = f"Vai trò: Chuyên gia {sector}. Chủ đề: {keyword}. Tone: {tone}. {extra_prompt}"
                    
                    response = model.generate_content(prompt)
                    st.session_state.result = response.text
                    st.session_state.type = content_type
                    st.session_state.kw = keyword
                    st.success("Xong!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    if 'result' in st.session_state:
        # 1. WEBSITE
        if st.session_state.type == "Bài Website chuẩn SEO":
            # Hiển thị Ảnh Featured (Luôn hiện đầu tiên)
            st.info("🖼️ Ảnh Featured (Ảnh bìa bài viết)")
            feat_prompt = f"{st.session_state.kw} insurance concept header"
            st.image(get_image_url(feat_prompt, 1200, 628), use_container_width=True)
            
            st.markdown("---")
            # Hiển thị nội dung + Ảnh inline
            render_mixed_content(st.session_state.result)
            
        # 2. FACEBOOK
        elif st.session_state.type == "Bài Facebook Viral":
            st.info("📱 Ảnh Facebook (Vuông)")
            fb_prompt = f"{st.session_state.kw} insurance flat lay square"
            st.image(get_image_url(fb_prompt, 1080, 1080), width=400)
            st.code(st.session_state.result, language='markdown')
            
        # 3. VIDEO
        else:
            tab1, tab2 = st.tabs(["🎬 Video", "📝 Kịch bản"])
            with tab1:
                if st.button("🎥 Dựng Video"):
                    with st.spinner("Đang render..."):
                        v = create_video_from_script(st.session_state.result)
                        if v: st.video(v)
            with tab2:
                st.text_area("Script", st.session_state.result, height=400)
