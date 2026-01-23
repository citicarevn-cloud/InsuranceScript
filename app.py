import streamlit as st
import time

# --- 1. CẤU HÌNH APP ---
st.set_page_config(page_title="Insurance Script Fix", layout="wide", page_icon="🛠️")

# --- 2. KIỂM TRA MÔI TRƯỜNG (DIAGNOSTIC) ---
st.title("🛠️ Chẩn đoán hệ thống Video")

try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    st.success(f"✅ Đã tìm thấy FFmpeg tại: `{ffmpeg_path}`")
except Exception as e:
    st.error(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy FFmpeg. Bạn đã tạo file `packages.txt` trên GitHub chưa?")
    st.stop()

try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip
    import google.generativeai as genai
    import requests
    import re
    import tempfile
    import os
    import asyncio
    import edge_tts
    import concurrent.futures
except ImportError as e:
    st.error(f"❌ Thiếu thư viện Python: {e}. Hãy kiểm tra `requirements.txt`.")
    st.stop()

# --- CSS ---
st.markdown("""
    <style>
    .stButton>button {background-color: #FF4B4B; color: white; font-weight: bold; width: 100%;}
    </style>
""", unsafe_allow_html=True)

# --- CONFIG VARIABLES ---
if 'feedback_history' not in st.session_state: st.session_state.feedback_history = []
VOICE_MAP = {
    "Chuyên nghiệp": "mJLZ5p8I7Pk81BHpKwbx", "Đời thường": "foH7s9fX31wFFH2yqrFa",
    "Cảm động": "1l0C0QA9c9jN22EmWiB0", "Hài hước": "JxmKvRaNYFidf0N27Vng"
}

# --- FUNCTIONS ---
def clean_text(text):
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    for p in ["Lời bình:", "Audio:", "Voice:", "Scene \d+:", "MC:", "Host:"]:
        text = re.sub(f'{p}', '', text, flags=re.IGNORECASE)
    return text.replace("*", "").replace("#", "").strip()

async def gen_edge_tts(text, voice, fname):
    await edge_tts.Communicate(text, voice).save(fname)

def gen_audio(text, fname, tone, provider, api_key, ms_voice):
    text = clean_text(text)
    if not text: return False
    
    if "ElevenLabs" in provider:
        if not api_key: st.error("❌ Thiếu ElevenLabs Key"); return False
        vid = VOICE_MAP.get(tone, "mJLZ5p8I7Pk81BHpKwbx")
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
            headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
            res = requests.post(url, json={"text": text, "model_id": "eleven_turbo_v2"}, headers=headers, timeout=60)
            if res.status_code == 200:
                with open(fname, 'wb') as f: f.write(res.content)
                return True
            else: st.error(f"❌ ElevenLabs Error: {res.text}"); return False
        except Exception as e: st.error(f"❌ Lỗi mạng: {e}"); return False

    if "Microsoft" in provider:
        try:
            asyncio.run(gen_edge_tts(text, ms_voice, fname))
            return True
        except Exception as e: st.error(f"❌ Microsoft TTS Error: {e}"); return False
    return False

def gen_image_hf(prompt, token, w, h):
    if not token: return None
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {token}"}
    full_prompt = prompt + ", high quality illustration, isometric, no text"
    full_prompt += ", vertical 9:16" if w < h else ", wide 16:9"
    for _ in range(3):
        try:
            res = requests.post(API_URL, headers=headers, json={"inputs": full_prompt}, timeout=20)
            if res.status_code == 200: return res.content
            time.sleep(1)
        except: time.sleep(1)
    return None

def process_scene(args):
    # Unpack arguments
    line, w, h, tone, provider, el_key, ms_voice, hf_token = args
    
    if "|" not in line: return None
    parts = line.split("|")
    if len(parts) < 2: return None
    
    img_p = parts[0].replace("Scene", "").replace(":", "").strip()
    aud_t = parts[1].strip()
    
    # Audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: af = f.name
    if not gen_audio(aud_t, af, tone, provider, el_key, ms_voice): return None
    
    # Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        data = gen_image_hf(img_p, hf_token, w, h)
        if data: f.write(data); img_path = f.name
        else: img_path = "PLACEHOLDER"
            
    return (af, img_path)

def create_video(script, w, h, tone, provider, el_key, ms_voice, hf_token):
    # Lọc script
    lines = [l for l in script.split('\n') if "|" in l and ("Scene" in l or "Cảnh" in l)][:10]
    if not lines:
        st.error("⚠️ Lỗi Kịch bản: Không tìm thấy dòng nào chứa '|' và 'Scene'.")
        return None
    
    progress_bar = st.progress(0)
    status_box = st.empty()
    status_box.info(f"🚀 Bắt đầu xử lý {len(lines)} cảnh...")
    
    # Chuẩn bị tham số
    args = [(l, w, h, tone, provider, el_key, ms_voice, hf_token) for l in lines]
    
    results = []
    # CHẠY TUẦN TỰ (Sequential) để dễ debug
    for i, arg in enumerate(args):
        status_box.text(f"⏳ Đang xử lý cảnh {i+1}/{len(lines)}...")
        res = process_scene(arg)
        results.append(res)
        progress_bar.progress((i+1)/len(lines))
    
    status_box.info("🎬 Đang ghép Video (Render)...")
    clips = []
    for res in results:
        if res:
            af, imgf = res
            try:
                ac = AudioFileClip(af)
                dur = ac.duration + 0.5
                if imgf == "PLACEHOLDER":
                    clip = ColorClip(size=(w, h), color=(0,0,0), duration=dur)
                else:
                    clip = ImageClip(imgf).set_duration(dur)
                
                clip = clip.set_audio(ac).set_fps(15) # FPS thấp để nhanh
                clips.append(clip)
            except Exception as e:
                st.warning(f"Bỏ qua 1 clip lỗi: {e}")

    if clips:
        try:
            final = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                # Dùng codec libx264 rõ ràng
                final.write_videofile(f.name, codec='libx264', audio_codec='aac', fps=15, preset='ultrafast', threads=4)
            status_box.success("✅ Xong!")
            return f.name
        except Exception as e:
            st.error(f"❌ Lỗi Render Final: {e}")
            return None
    else:
        st.error("❌ Không tạo được clip nào.")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Cấu hình")
    
    # Secrets
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    eleven_api = st.secrets.get("ELEVEN_API_KEY", "").strip()
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "").strip()

    if not api_key: st.error("Thiếu Gemini Key")
    if not hf_token: st.error("Thiếu HF Token")

    tts_provider = st.selectbox("Server Giọng:", ["ElevenLabs (VIP)", "Microsoft (Free)"])
    edge_voice = "vi-VN-HoaiMyNeural"
    if "Microsoft" in tts_provider:
        edge_voice = st.selectbox("Giọng MS:", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]).split(" ")[0]

    st.divider()
    
    # TEST SYSTEM BUTTON
    if st.button("🛠️ Test Hệ Thống Video"):
        with st.spinner("Đang test FFmpeg..."):
            try:
                # Tạo video giả 3s
                clip = ColorClip(size=(640, 360), color=(255,0,0), duration=3)
                clip.fps = 15
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    clip.write_videofile(f.name, codec='libx264', preset='ultrafast')
                st.success("✅ Hệ thống Video hoạt động tốt!")
                st.video(f.name)
            except Exception as e:
                st.error(f"❌ Lỗi Hệ Thống: {e}")
                st.error("👉 Bạn chưa cài FFmpeg hoặc packages.txt sai.")

# --- MAIN ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Nhập liệu")
    kw = st.text_input("Từ khóa", "Bảo hiểm nhân thọ")
    pillar = st.selectbox("Pillar", ["Kiến thức", "Sản phẩm", "Niềm tin"])
    angle = st.selectbox("Angle", ["Chuyên gia", "Storytelling", "Hài hước"])
    
    vw, vh = 720, 1280
    if st.radio("Khung hình", ["Dọc 9:16", "Ngang 16:9"]) == "Ngang 16:9": vw, vh = 1280, 720
    
    tone_map = {"Chuyên gia": "Chuyên nghiệp", "Storytelling": "Cảm động", "Hài hước": "Hài hước"}
    auto_tone = tone_map.get(angle, "Chuyên nghiệp")
    
    if st.button("🚀 VIẾT KỊCH BẢN"):
        if not api_key: st.error("Thiếu Key")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            prompt = f"""
            Topic: {kw}. Pillar: {pillar}. Angle: {angle}.
            Output:
            1. Title
            2. Hashtags
            3. Script Video 45s.
            IMPORTANT: Each scene MUST follow this format exactly:
            Scene X: [English visual prompt] | [Vietnamese audio script]
            """
            with st.spinner("Đang viết..."):
                res = model.generate_content(prompt)
                st.session_state.res = res.text
                st.session_state.w = vw
                st.session_state.h = vh
                st.session_state.tone = auto_tone

with col2:
    st.subheader("2. Kết quả")
    if 'res' in st.session_state:
        r = st.session_state.res
        w = st.session_state.w
        h = st.session_state.h
        t = st.session_state.tone
        
        st.text_area("Script", r, height=300)
        
        if st.button("🎥 Dựng Video (Debug Mode)"):
            v = create_video(r, w, h, t, tts_provider, eleven_api, edge_voice, hf_token)
            if v: st.video(v)
