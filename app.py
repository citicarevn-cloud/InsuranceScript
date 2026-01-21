import streamlit as st
import google.generativeai as genai
# gTTS đã bị loại bỏ hoàn toàn để không bao giờ ra giọng Google
import tempfile
import os
import requests
import re
import time
import random
import asyncio
import edge_tts
import concurrent.futures
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip

# --- 1. CẤU HÌNH APP (ĐÃ ĐỔI TÊN) ---
st.set_page_config(page_title="Insurance Script", layout="wide", page_icon="🛡️")

# --- CSS GIAO DIỆN ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%;}
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    .stProgress > div > div > div > div { background-color: #28a745; }
    /* Ẩn bớt các element thừa */
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- QUẢN LÝ TRẠNG THÁI ---
if 'feedback_history' not in st.session_state: st.session_state.feedback_history = []
if 'video_settings' not in st.session_state: st.session_state.video_settings = {'w': 1280, 'h': 720}

# --- CẤU HÌNH VOICE ID ---
VOICE_MAP = {
    "Chuyên nghiệp": "mJLZ5p8I7Pk81BHpKwbx",  # Nam Sadoma
    "Đời thường": "foH7s9fX31wFFH2yqrFa",     # Huyen
    "Cảm động": "1l0C0QA9c9jN22EmWiB0",       # Jade
    "Hài hước": "JxmKvRaNYFidf0N27Vng"        # Son Tran
}

# --- SIDEBAR (BẢNG ĐIỀU KHIỂN) ---
with st.sidebar:
    st.header("🎛️ Cấu hình hệ thống")
    
    if st.button("🔄 LÀM MỚI (RESET)"):
        st.session_state.clear()
        st.rerun()

    # 1. NHẬP KEY (TỰ ĐỘNG LẤY TỪ SECRETS)
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    eleven_api = st.secrets.get("ELEVEN_API_KEY", "").strip()
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "").strip()

    # Kiểm tra trạng thái Key
    if api_key: st.success("✅ Gemini: Sẵn sàng")
    else: st.error("❌ Thiếu Gemini Key")
        
    if eleven_api: st.success("✅ ElevenLabs: Sẵn sàng")
    else: st.warning("⚠️ Thiếu ElevenLabs Key (Sẽ dùng Microsoft)")
        
    if hf_token: st.success("✅ HuggingFace: Sẵn sàng")
    else: st.error("❌ Thiếu HuggingFace Token (Không thể tạo ảnh)")

    st.divider()
    
    # 2. CHỌN MODEL GEMINI
    st.subheader("🧠 Bộ não xử lý")
    available_models = ["models/gemini-1.5-flash"] # Mặc định nhanh
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except: pass
    selected_model = st.selectbox("Chọn Model:", available_models, index=0)

    st.divider()

    # 3. CHỌN GIỌNG ĐỌC (KHÔNG CÓ GOOGLE)
    st.subheader("🔊 Nguồn giọng đọc")
    tts_provider = st.selectbox("Chọn Server:", ["ElevenLabs (VIP)", "Microsoft (Miễn phí)"])
    
    edge_voice = "vi-VN-HoaiMyNeural" 
    if "Microsoft" in tts_provider:
        edge_voice = st.selectbox("Giọng Microsoft:", [
            "vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"
        ]).split(" ")[0]

# --- HÀM XỬ LÝ TEXT ---
def clean_text_for_audio(text):
    # Loại bỏ chỉ dẫn cảnh, lời bình, ký tự đặc biệt
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    prefixes = ["Lời bình:", "Audio:", "Voice:", "Thuyết minh:", "Host:", "MC:", "Scene \d+:"]
    for p in prefixes:
        text = re.sub(f'{p}', '', text, flags=re.IGNORECASE)
    text = text.replace("*", "").replace("#", "").replace("- ", "").replace('"', '')
    return text.strip()

# --- HÀM TẠO AUDIO (STRICT MODE - KHÔNG GOOGLE) ---
async def generate_edge_tts(text, voice, filename):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

def generate_audio_strict(text, filename, tone_key="Chuyên nghiệp"):
    clean_text = clean_text_for_audio(text)
    if not clean_text: return False
    
    # 1. ELEVENLABS
    if "ElevenLabs" in tts_provider:
        if not eleven_api:
            st.error("❌ Bạn chọn ElevenLabs nhưng chưa có Key!")
            return False
            
        voice_id = VOICE_MAP.get(tone_key, "mJLZ5p8I7Pk81BHpKwbx")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": eleven_api, "Content-Type": "application/json"}
        # Dùng model Turbo cho nhanh
        data = {"text": clean_text, "model_id": "eleven_turbo_v2"} 
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)
            if response.status_code == 200:
                with open(filename, 'wb') as f: f.write(response.content)
                return True
            else:
                # Báo lỗi rõ ràng
                st.error(f"❌ ElevenLabs Lỗi {response.status_code}: {response.text}")
                return False 
        except Exception as e:
            st.error(f"❌ Lỗi mạng ElevenLabs: {e}")
            return False

    # 2. MICROSOFT
    if "Microsoft" in tts_provider:
        try:
            asyncio.run(generate_edge_tts(clean_text, edge_voice, filename))
            return True
        except Exception as e:
            st.error(f"❌ Lỗi Microsoft TTS: {e}")
            return False

    return False

# --- HÀM TẠO ẢNH (HUGGING FACE ONLY - KHÔNG POLLINATIONS) ---
def generate_image_hf_only(prompt, width, height):
    if not hf_token: return None

    # Model SDXL chuẩn
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    style = ", high quality illustration, isometric style, flat design, cinematic lighting, no text"
    full_prompt = prompt + style
    if width < height: full_prompt += ", vertical, 9:16 portrait"
    else: full_prompt += ", wide angle, 16:9 landscape"

    # Thử 3 lần
    for i in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": full_prompt}, timeout=25)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 503: # Model đang load
                time.sleep(2)
            else:
                time.sleep(1)
        except: time.sleep(1)
    return None

# --- HÀM DỰNG SCENE ---
def process_scene_strict(args):
    part, width, height, tone = args
    try:
        if "|" in part:
            data = part.split("|")
            if len(data) < 2: return None
            
            img_prompt = data[0].replace("Scene", "").replace(":", "").strip()
            raw_voice_text = data[1].strip()
            
            # 1. Audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                audio_path = f.name
            
            if not generate_audio_strict(raw_voice_text, audio_path, tone):
                return None # Không có tiếng thì bỏ qua luôn

            # 2. Ảnh (Chỉ HF)
            img_content = generate_image_hf_only(img_prompt, width, height)
            
            if img_content:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(img_content); img_path = f.name
                return (audio_path, img_path)
            else:
                return (audio_path, "PLACEHOLDER")
    except: return None

def create_video_strict(script_data, width, height, tone):
    lines = [line for line in script_data.strip().split('\n') if "|" in line and "Scene" in line]
    if len(lines) > 10: lines = lines[:10] # Max 10 cảnh
    
    total = len(lines)
    if total == 0: return None

    bar = st.progress(0)
    st.caption("🚀 Đang xử lý tài nguyên (Tuần tự để an toàn)...")
    
    args = [(line, width, height, tone) for line in lines]
    results = []
    
    # Chạy tuần tự 1 luồng
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for i, res in enumerate(executor.map(process_scene_strict, args)):
            results.append(res)
            bar.progress(int((i+1)/total * 100))
            
    st.caption("🎬 Đang ghép video...")
    clips = []
    for asset in results:
        if asset:
            apath, ipath = asset
            try:
                ac = AudioFileClip(apath)
                if ipath == "PLACEHOLDER":
                    # Màn hình đen nếu ảnh lỗi
                    clip = ColorClip(size=(width, height), color=(0,0,0), duration=ac.duration+0.5)
                else:
                    clip = ImageClip(ipath).set_duration(ac.duration+0.5)
                
                clip = clip.set_audio(ac).set_fps(15)
                clips.append(clip)
            except: pass

    if clips:
        final = concatenate_videoclips(clips, method="compose")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            final.write_videofile(f.name, codec='libx264', audio_codec='aac', fps=15, preset='ultrafast', threads=4)
        bar.empty()
        return f.name
    return None

def render_mixed_content(text, width=800, height=450):
    parts = re.split(r'\{{1,2}IMAGE:?\s*(.*?)\}{1,2}', text, flags=re.IGNORECASE)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip(): st.markdown(part)
        else:
            prompt = part.strip().replace("}", "").replace("{", "")
            if prompt:
                data = generate_image_hf_only(prompt, width, height)
                if data: st.image(data, caption=f"🎨 {prompt}", use_container_width=True)

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ Insurance Script") # Đã đổi tên
