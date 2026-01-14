import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
import re
import time
import random
import asyncio
import edge_tts
import concurrent.futures
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media AI Studio", layout="wide", page_icon="🎙️")

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%;}
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    .stProgress > div > div > div > div { background-color: #28a745; }
    </style>
""", unsafe_allow_html=True)

# --- QUẢN LÝ SESSION ---
if 'feedback_history' not in st.session_state: st.session_state.feedback_history = []
if 'video_settings' not in st.session_state: st.session_state.video_settings = {'w': 1280, 'h': 720}

# --- CẤU HÌNH VOICE ID (CỐ ĐỊNH - GIỮ NGUYÊN) ---
VOICE_MAP = {
    "Chuyên nghiệp": "mJLZ5p8I7Pk81BHpKwbx",  # Nam Sadoma
    "Đời thường": "foH7s9fX31wFFH2yqrFa",     # Huyen
    "Cảm động": "1l0C0QA9c9jN22EmWiB0",       # Jade
    "Hài hước": "JxmKvRaNYFidf0N27Vng"        # Son Tran
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Cấu hình hệ thống")
    
    if st.button("🔄 LÀM MỚI (RESET)"):
        saved = st.session_state.feedback_history
        st.session_state.clear()
        st.session_state.feedback_history = saved
        st.rerun()

    # 1. API KEY (GIỮ NGUYÊN)
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    eleven_api = st.secrets.get("ELEVEN_API_KEY", "")

    if api_key:
        st.success(f"✅ Gemini API: Đã kết nối")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
        
    if eleven_api:
        st.success(f"✅ ElevenLabs API: Đã kết nối")
    else:
        eleven_api = st.text_input("ElevenLabs API Key", type="password")

    st.divider()
    
    # 2. CHỌN MODEL (ĐÃ KHÔI PHỤC TÍNH NĂNG QUÉT)
    st.subheader("🧠 Bộ não xử lý")
    
    # Logic quét model tự động
    available_models = ["models/gemini-pro"] # Mặc định an toàn
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Lấy danh sách thực tế từ Project của bạn
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except Exception as e:
            st.error(f"Không quét được model: {e}")
            
    # Cho phép người dùng chọn từ danh sách đã quét
    selected_model = st.selectbox("Chọn Model:", available_models, index=0)

    st.divider()

    # 3. CẤU HÌNH GIỌNG ĐỌC (GIỮ NGUYÊN)
    st.subheader("🔊 Nguồn giọng đọc")
    tts_provider = st.selectbox("Chọn Server:", ["ElevenLabs (VIP - Nên dùng)", "Microsoft (Miễn phí)", "Google (Cơ bản)"])
    
    edge_voice = "vi-VN-HoaiMyNeural" 
    if "Microsoft" in tts_provider:
        edge_voice = st.selectbox("Chọn giọng MS:", [
            "vi-VN-HoaiMyNeural (Nữ - Truyền cảm)", 
            "vi-VN-NamMinhNeural (Nam - Trầm ấm)"
        ]).split(" ")[0]

# --- HÀM XỬ LÝ (GIỮ NGUYÊN CÁC CHỨC NĂNG TỐT) ---

def clean_text_for_audio(text):
    """Làm sạch văn bản"""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    prefixes = ["Lời bình:", "Audio:", "Voice:", "Thuyết minh:", "Host:", "MC:", "Scene \d+:"]
    for p in prefixes:
        text = re.sub(f'{p}', '', text, flags=re.IGNORECASE)
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', '')
    return text.strip()

async def generate_edge_tts(text, voice, filename):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

def generate_audio_unified(text, filename, tone_key="Chuyên nghiệp"):
    clean_text = clean_text_for_audio(text)
    if not clean_text: return False
    
    # ElevenLabs
    if "ElevenLabs" in tts_provider and eleven_api:
        voice_id = VOICE_MAP.get(tone_key, "mJLZ5p8I7Pk81BHpKwbx") 
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": eleven_api, "Content-Type": "application/json"}
        data = {"text": clean_text, "model_id": "eleven_multilingual_v2"}
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                with open(filename, 'wb') as f: f.write(response.content)
                return True
        except: pass 
        
    # Microsoft Edge TTS
    if "Microsoft" in tts_provider:
        try:
            asyncio.run(generate_edge_tts(clean_text, edge_voice, filename))
            return True
        except: pass

    # Google TTS
    try:
        tts = gTTS(text=clean_text, lang='vi')
        tts.save(filename)
        return True
    except: return False

def get_image_url(prompt, width=1280, height=720):
    # Giữ nguyên delay và random seed để tránh rate limit
    time.sleep(random.uniform(1.0, 3.0)) 
    seed = random.randint(1, 10000000)
    ratio_prompt = ", vertical, tall, 9:16" if width < height else ", wide angle, cinematic, 16:9"
    style = ", high quality illustration, isometric style, flat design, cinematic lighting, no text"
    clean_prompt = (prompt + style + ratio_prompt).replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&nologo=true&seed={seed}"

def process_scene(args):
    part, width, height, tone = args
    try:
        if "|" in part:
            data = part.split("|")
            if len(data) < 2: return None
            
            img_prompt = data[0].replace("Scene", "").replace(":", "").strip()
            raw_voice_text = data[1].strip()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                audio_path = f.name
            
            success = generate_audio_unified(raw_voice_text, audio_path, tone)
            if not success: return None

            img_url = get_image_url(img_prompt, width, height)
            response = requests.get(img_url, timeout=20)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(response.content); img_path = f.name
                return (audio_path, img_path)
            else: return None
    except: return None

def create_video_from_script(script_data, width, height, tone):
    lines = [line for line in script_data.strip().split('\n') if "|" in line and "Scene" in line]
    if len(lines) > 10: lines = lines[:10]
    total_scenes = len(lines)
    if total_scenes == 0: return None

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"🚀 Đang tải tài nguyên (Tone: {tone})...")
    
    process_args = [(line, width, height, tone) for line in lines]
    
    # Đa luồng (Max 2 workers để an toàn ảnh)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_scene, process_args))
        
    status_text.text("🎬 Đang render video...")
    clips = []
    for i, asset in enumerate(results):
        if asset:
            audio_path, img_path = asset
            try:
                ac = AudioFileClip(audio_path)
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
        except: return None
    return None

def render_mixed_content(text, width=800, height=450):
    pattern = r'\{{1,2}IMAGE:?\s*(.*?)\}{1,2}'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip(): st.markdown(part)
        else:
            img_prompt = part.strip().replace("}", "").replace("{", "")
            if img_prompt:
                img_url = get_image_url(img_prompt, width, height)
                st.image(img_url, caption=f"🎨 {img_prompt}", use_container_width=True)

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ DAT Media AI Studio")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Thiết lập nội dung")
    keyword = st.text_input("Chủ đề chính", "Bảo hiểm nhân thọ cho người trụ cột")
    sector = st.selectbox("Lĩnh vực", ["Bảo hiểm Nhân thọ", "Bảo hiểm Phi Nhân thọ", "Bảo hiểm Sức khoẻ"])
    content_type = st.radio("Loại nội dung", ["Clip (Video)", "Bài Website", "Bài Facebook"])
    
    seo_guide = ""
    video_w, video_h = 1280, 720
    
    if content_type == "Clip (Video)":
        orientation = st.radio("Khung hình:", ["Ngang 16:9 (YouTube)", "Dọc 9:16 (TikTok/Shorts)"], horizontal=True)
        if "Ngang" in orientation:
            video_w, video_h = 1280, 720; ratio_txt = "Wide 16:9"
        else:
            video_w, video_h = 720, 1280; ratio_txt = "Vertical 9:16"

        vid_len = st.radio("Độ dài:", ["Clip Ngắn (<90s)", "Video Dài (Preview)"], horizontal=True)
        if "Ngắn" in vid_len: dur = st.slider("Giây", 15, 90, 60); dur_txt = f"{dur} giây"
        else: dur = st.slider("Phút", 2, 20, 5); dur_txt = f"{dur} phút"

        seo_guide = f"""
        - Viết Kịch bản Video ({ratio_txt}) dài {dur_txt}.
        - Định dạng BẮT BUỘC từng dòng: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'.
        """
    elif content_type == "Bài Website":
        words = st.number_input("Số từ", 500, 2500, 1000)
        seo_guide = f"- Viết bài chuẩn SEO {words} từ. BẮT BUỘC dùng thẻ {{IMAGE: english prompt}} xen kẽ."
    else:
        seo_guide = "- Viết Caption Facebook thu hút. Đề xuất ảnh vuông."

    tone_options = ["Chuyên nghiệp", "Đời thường", "Cảm động", "Hài hước"]
    tone = st.select_slider("Tone giọng & Phong cách", tone_options)
    
    btn_run = st.button("🚀 XỬ LÝ NGAY")

# --- KẾT QUẢ ---
with col2:
    st.subheader("2. Kết quả")
    if btn_run:
        if not api_key: st.error("Chưa kết nối Gemini API")
        else:
            with st.spinner(f"AI đang quét model và xử lý..."):
                try:
                    # Lưu cài đặt video
                    st.session_state.video_settings = {'w': video_w, 'h': video_h}
                    st.session_state.tone_key = tone
                    
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(selected_model) # Sử dụng Model từ danh sách quét
                    
                    past_fb = "\n".join([f"- {fb}" for fb in st.session_state.feedback_history])
                    prompt = f"""
                    Chủ đề: {keyword}. Lĩnh vực: {sector}. Tone: {tone}.
                    YÊU CẦU ĐẦU RA:
                    1. TIÊU ĐỀ CHUẨN SEO
                    2. 5 HASHTAGS & 5 TAGS
                    3. NỘI DUNG: {seo_guide}
                    LƯU Ý: Không dùng dấu ** trong lời bình.
                    LƯU Ý USER: {past_fb}
                    """
                    response = model.generate_content(prompt)
                    st.session_state.result = response.text
                    st.session_state.type = content_type
                    st.session_state.kw = keyword
                    st.success("Đã có nội dung!")
                except Exception as e: st.error(f"Lỗi: {e}. Hãy thử chọn Model khác ở Sidebar.")

    if 'result' in st.session_state:
        if st.session_state.type == "Bài Website":
            st.image(get_image_url(f"{st.session_state.kw} insurance header", 1200, 628), use_container_width=True)
            render_mixed_content(st.session_state.result)
        elif st.session_state.type == "Bài Facebook":
            st.image(get_image_url(f"{st.session_state.kw} flat lay", 1080, 1080), width=450)
            st.markdown(st.session_state.result)
        else:
            tab1, tab2 = st.tabs(["🎬 Video Demo", "📝 Kịch bản"])
            with tab1:
                vw = st.session_state.video_settings['w']
                vh = st.session_state.video_settings['h']
                tk = st.session_state.get('tone_key', "Chuyên nghiệp")
                
                # Hiển thị thông tin giọng đang dùng
                voice_name_map = {"mJLZ5p8I7Pk81BHpKwbx": "Nam Sadoma", "foH7s9fX31wFFH2yqrFa": "Huyền", "1l0C0QA9c9jN22EmWiB0": "Jade", "JxmKvRaNYFidf0N27Vng": "Sơn Trần"}
                current_id = VOICE_MAP.get(tk, "")
                v_label = voice_name_map.get(current_id, "Mặc định")
                
                if "ElevenLabs" in tts_provider:
                    st.info(f"🎙️ Đang dùng giọng: **{v_label}** (Tone: {tk})")
                
                if st.button("🎥 Dựng Video"):
                    v_path = create_video_from_script(st.session_state.result, vw, vh, tk)
                    if v_path: st.video(v_path)
            with tab2:
                st.text_area("Script", st.session_state.result, height=500)
