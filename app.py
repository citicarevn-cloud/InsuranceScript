import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
import re
import time
import random
import concurrent.futures
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media AI Studio", layout="wide", page_icon="🎬")

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%;}
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    /* Tùy chỉnh thanh tiến trình */
    .stProgress > div > div > div > div { background-color: #FF4B4B; }
    </style>
""", unsafe_allow_html=True)

# --- QUẢN LÝ SESSION ---
if 'feedback_history' not in st.session_state: st.session_state.feedback_history = []
if 'video_settings' not in st.session_state: st.session_state.video_settings = {'w': 1280, 'h': 720}

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Bảng Điều Khiển")
    
    if st.button("🔄 LÀM MỚI (RESET)"):
        saved = st.session_state.feedback_history
        st.session_state.clear()
        st.session_state.feedback_history = saved
        st.rerun()

    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Đã kết nối API")
    else:
        api_key = st.text_input("Nhập API Key", type="password")

    # Chọn Model
    available_models = ["models/gemini-1.5-flash", "models/gemini-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except: pass
    selected_model = st.selectbox("Model:", available_models, index=0)
    
    with st.expander(f"🧠 Trí nhớ AI ({len(st.session_state.feedback_history)})"):
        for fb in st.session_state.feedback_history: st.text(f"- {fb}")
        if st.button("Xóa trí nhớ"):
            st.session_state.feedback_history = []
            st.rerun()

# --- HÀM XỬ LÝ (BACKEND) ---

def get_image_url(prompt, width=1280, height=720):
    """Tạo URL ảnh với kích thước động"""
    seed = random.randint(1, 999999)
    # Thêm từ khóa định hướng khung hình
    ratio_prompt = ", vertical, tall, portrait" if width < height else ", wide angle, cinematic, horizontal"
    style = ", high quality illustration, isometric style, flat design, cinematic lighting, no text"
    
    clean_prompt = (prompt + style + ratio_prompt).replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&nologo=true&seed={seed}"

def process_scene(args):
    """Xử lý song song (Nhận tuple args để tương thích map)"""
    part, width, height = args
    try:
        if "|" in part:
            data = part.split("|")
            if len(data) < 2: return None
            
            img_prompt = data[0].replace("Scene", "").replace(":", "").strip()
            voice_text = data[1].strip()
            
            # 1. Tạo Audio
            tts = gTTS(text=voice_text, lang='vi')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name); audio_path = f.name
            
            # 2. Tải ảnh (đúng kích thước 16:9 hoặc 9:16)
            img_url = get_image_url(img_prompt, width, height)
            response = requests.get(img_url, timeout=15)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(response.content); img_path = f.name
            
            return (audio_path, img_path)
    except Exception as e:
        return None

def create_video_from_script(script_data, width, height):
    """Dựng video ĐA LUỒNG"""
    clips = []
    # Lọc lấy các dòng Scene
    lines = [line for line in script_data.strip().split('\n') if "|" in line and "Scene" in line]
    
    # GIỚI HẠN RENDER CHO VIDEO DÀI (Tránh sập server)
    if len(lines) > 15:
        st.warning(f"⚠️ Kịch bản rất dài ({len(lines)} cảnh). Để tránh sập server, AI sẽ chỉ dựng bản Demo 15 cảnh đầu tiên.")
        lines = lines[:15]

    total_scenes = len(lines)
    if total_scenes == 0: return None

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Tải tài nguyên (Kèm kích thước w, h)
    status_text.text(f"🚀 Đang tải tài nguyên ({width}x{height})...")
    
    # Chuẩn bị tham số cho hàm map
    process_args = [(line, width, height) for line in lines]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_scene, process_args))
        
    # 2. Dựng Clip
    status_text.text("🎬 Đang render video...")
    for i, asset in enumerate(results):
        if asset:
            audio_path, img_path = asset
            try:
                ac = AudioFileClip(audio_path)
                # FPS 15 để render nhanh
                clip = ImageClip(img_path).set_duration(ac.duration + 0.5).set_audio(ac).set_fps(15)
                clips.append(clip)
            except: pass
        progress_bar.progress(int((i + 1) / total_scenes * 100))

    if clips:
        try:
            final = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                final.write_videofile(f.name, codec='libx264', audio_codec='aac', fps=15, preset='ultrafast', threads=4)
            status_text.text("✅ Xong!")
            progress_bar.empty()
            return f.name
        except Exception as e:
            st.error(f"Lỗi render: {e}")
            return None
    return None

def render_mixed_content(text, width=800, height=450):
    """Hiển thị bài viết kèm ảnh"""
    pattern = r'\{{1,2}IMAGE:?\s*(.*?)\}{1,2}'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip(): st.markdown(part)
        else:
            img_prompt = part.strip().replace("}", "").replace("{", "")
            if img_prompt:
                # Ảnh trong bài viết thì giữ tỷ lệ chữ nhật ngang cho dễ nhìn
                img_url = get_image_url(img_prompt, width, height)
                st.image(img_url, caption=f"🎨 {img_prompt}", use_container_width=True)

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ AI Content Generator: Đa Nền Tảng")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Thiết lập nội dung")
    keyword = st.text_input("Chủ đề chính", "Bảo hiểm du lịch quốc tế")
    sector = st.selectbox("Lĩnh vực", ["Nhân thọ", "Phi nhân thọ", "Sức khỏe", "Tài chính"])
    
    # CHỌN LOẠI NỘI DUNG
    content_type = st.radio("Loại nội dung", ["Clip (Video)", "Bài Website", "Bài Facebook"])
    
    # --- LOGIC CẤU HÌNH CHI TIẾT ---
    seo_guide = ""
    video_w, video_h = 1280, 720 # Default
    
    if content_type == "Clip (Video)":
        # 1. Chọn hướng video (Yêu cầu mới)
        orientation = st.radio("Khung hình:", ["Ngang 16:9 (YouTube)", "Dọc 9:16 (TikTok/Shorts)"], horizontal=True)
        if "Ngang" in orientation:
            video_w, video_h = 1280, 720
            ratio_txt = "Wide 16:9"
        else:
            video_w, video_h = 720, 1280
            ratio_txt = "Vertical 9:16"

        # 2. Chọn thời lượng (Yêu cầu mới)
        vid_len_type = st.radio("Độ dài:", ["Clip Ngắn (<90s)", "Video Dài (tối đa 20')"], horizontal=True)
        
        if "Ngắn" in vid_len_type:
            duration_val = st.slider("Thời lượng (Giây)", 15, 90, 60)
            duration_txt = f"{duration_val} giây"
            platform = "TikTok/Reels/Shorts"
        else:
            duration_min = st.slider("Thời lượng (Phút)", 2, 20, 5)
            duration_txt = f"{duration_min} phút"
            platform = "YouTube Long-form"
            st.info("💡 Lưu ý: Với video dài, AI sẽ viết kịch bản full, nhưng nút 'Dựng Video' sẽ chỉ tạo bản Preview khoảng 1-2 phút đầu.")

        seo_guide = f"""
        - Vai trò: Nhà sáng tạo nội dung {platform}.
        - Nhiệm vụ: Viết Kịch bản Video ({ratio_txt}) dài khoảng {duration_txt}.
        - Cấu trúc: Chia thành nhiều Scene. Mỗi dòng BẮT BUỘC định dạng: 'Scene X: [Mô tả hình ảnh tiếng Anh chi tiết] | [Lời bình tiếng Việt]'.
        - Yêu cầu hình ảnh: Phải mô tả rõ góc máy ({ratio_txt}) để AI vẽ đúng khung hình.
        """
        
    elif content_type == "Bài Website":
        platform = "Google Search"
        words = st.number_input("Số từ", 500, 3000, 1000)
        seo_guide = f"- Viết bài chuẩn SEO {words} từ. BẮT BUỘC dùng thẻ {{IMAGE: english prompt}} để chèn ảnh minh họa."
        
    else: # Facebook
        platform = "Facebook"
        seo_guide = "- Viết Caption thu hút, viral. Đề xuất ý tưởng ảnh vuông."

    tone = st.select_slider("Tone giọng", ["Hài hước", "Đời thường", "Chuyên nghiệp", "Cảm động"])
    btn_run = st.button("🚀 XỬ LÝ NGAY")

# --- KẾT QUẢ ---
with col2:
    st.subheader("2. Kết quả")
    
    if btn_run:
        if not api_key: st.error("Chưa nhập API Key")
        else:
            with st.spinner(f"AI đang làm việc ({platform})..."):
                try:
                    # Lưu cài đặt video vào session để dùng cho nút Render
                    st.session_state.video_settings = {'w': video_w, 'h': video_h}
                    
                    model = genai.GenerativeModel(selected_model)
                    past_fb = "\n".join([f"- {fb}" for fb in st.session_state.feedback_history])
                    
                    prompt = f"""
                    Chủ đề: {keyword}. Lĩnh vực: {sector}. Tone: {tone}.
                    
                    YÊU CẦU ĐẦU RA (BẮT BUỘC):
                    1. TIÊU ĐỀ CHUẨN SEO (Hấp dẫn, chứa từ khóa)
                    2. 5 HASHTAGS (#) & 5 TAGS (SEO)
                    3. NỘI DUNG CHÍNH:
                    {seo_guide}
                    
                    LƯU Ý TỪ QUÁ KHỨ: {past_fb}
                    """
                    response = model.generate_content(prompt)
                    st.session_state.result = response.text
                    st.session_state.type = content_type
                    st.session_state.kw = keyword
                    st.success("Xong!")
                except Exception as e: st.error(f"Lỗi: {e}")

    if 'result' in st.session_state:
        # A. WEBSITE
        if st.session_state.type == "Bài Website":
            st.info("🖼️ Ảnh Featured")
            st.image(get_image_url(f"{st.session_state.kw} insurance header", 1200, 628), use_container_width=True)
            render_mixed_content(st.session_state.result)
            
        # B. FACEBOOK
        elif st.session_state.type == "Bài Facebook":
            st.info("📱 Ảnh Vuông")
            st.image(get_image_url(f"{st.session_state.kw} flat lay", 1080, 1080), width=450)
            st.markdown(st.session_state.result)
            
        # C. VIDEO (XỬ LÝ ĐA KHUNG HÌNH)
        else:
            tab1, tab2 = st.tabs(["🎬 Video Demo", "📝 Kịch bản Chi tiết"])
            
            with tab1:
                # Lấy kích thước đã lưu
                vw = st.session_state.video_settings['w']
                vh = st.session_state.video_settings['h']
                
                st.caption(f"Đang cấu hình Render: {vw}x{vh} (Turbo Mode)")
                
                if st.button("🎥 Dựng Video Ngay"):
                    v_path = create_video_from_script(st.session_state.result, vw, vh)
                    if v_path: st.video(v_path)
            
            with tab2:
                st.text_area("Script", st.session_state.result, height=500)

        # FEEDBACK FORM
        st.markdown("---")
        with st.form("fb_form"):
            c1, c2 = st.columns([1,3])
            r = c1.slider("Đánh giá", 1, 5, 5)
            c = c2.text_input("Góp ý (AI sẽ ghi nhớ):")
            if st.form_submit_button("Gửi Feedback"):
                if c: st.session_state.feedback_history.append(f"{r} sao: {c}")
                st.success("Đã ghi nhận!")
