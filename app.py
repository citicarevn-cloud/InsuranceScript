import streamlit as st

# --- 1. CẤU HÌNH APP ---
st.set_page_config(page_title="Insurance Script", layout="wide", page_icon="🛡️")

# --- 2. KIỂM TRA THƯ VIỆN (DEBUG) ---
try:
    import edge_tts
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip, TextClip
    import google.generativeai as genai
    import requests
    import re
    import tempfile
    import os
    import time
    import asyncio
    import concurrent.futures
except ImportError as e:
    st.error(f"❌ LỖI THIẾU THƯ VIỆN: {e}")
    st.info("👉 Vui lòng vào file `requirements.txt` thêm dòng: `edge-tts` và `moviepy==1.0.3`")
    st.stop()

# --- 3. CSS GIAO DIỆN ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%;}
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# --- QUẢN LÝ TRẠNG THÁI ---
if 'feedback_history' not in st.session_state: st.session_state.feedback_history = []

# --- CẤU HÌNH VOICE ID ---
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
        st.session_state.clear()
        st.rerun()

    # Nhập Key (Tự động lấy từ Secrets hoặc nhập tay)
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    eleven_api = st.secrets.get("ELEVEN_API_KEY", "").strip()
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "").strip()

    if api_key: st.success("✅ Gemini: OK")
    else: st.error("❌ Thiếu Gemini Key")
    
    if eleven_api: st.success("✅ ElevenLabs: OK")
    else: st.warning("⚠️ Chưa có ElevenLabs (Sẽ dùng Microsoft)")

    if hf_token: st.success("✅ HuggingFace: OK")
    else: st.error("❌ Thiếu HuggingFace (Không thể tạo ảnh)")

    st.divider()
    
    # Chọn Model
    st.subheader("🧠 Bộ não xử lý")
    available_models = ["models/gemini-1.5-flash"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except: pass
    selected_model = st.selectbox("Model:", available_models, index=0)

    st.divider()

    # Chọn Giọng
    st.subheader("🔊 Nguồn giọng đọc")
    tts_provider = st.selectbox("Server:", ["ElevenLabs (VIP)", "Microsoft (Miễn phí)"])
    
    edge_voice = "vi-VN-HoaiMyNeural"
    if "Microsoft" in tts_provider:
        edge_voice = st.selectbox("Giọng MS:", ["vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"]).split(" ")[0]

# --- HÀM XỬ LÝ (CORE) ---
def clean_text(text):
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    for p in ["Lời bình:", "Audio:", "Voice:", "Scene \d+:", "MC:", "Host:"]:
        text = re.sub(f'{p}', '', text, flags=re.IGNORECASE)
    return text.replace("*", "").replace("#", "").replace("- ", "").replace('"', '').strip()

async def gen_edge_tts(text, voice, fname):
    await edge_tts.Communicate(text, voice).save(fname)

def gen_audio(text, fname, tone):
    text = clean_text(text)
    if not text: return False
    
    if "ElevenLabs" in tts_provider:
        if not eleven_api:
            st.error("❌ Chọn ElevenLabs nhưng thiếu Key!"); return False
        
        vid = VOICE_MAP.get(tone, "mJLZ5p8I7Pk81BHpKwbx")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
        headers = {"xi-api-key": eleven_api, "Content-Type": "application/json"}
        # Dùng model turbo cho nhanh
        try:
            res = requests.post(url, json={"text": text, "model_id": "eleven_turbo_v2"}, headers=headers, timeout=60)
            if res.status_code == 200:
                with open(fname, 'wb') as f: f.write(res.content)
                return True
            else:
                st.error(f"❌ Lỗi ElevenLabs: {res.status_code}"); return False
        except: return False

    if "Microsoft" in tts_provider:
        try:
            asyncio.run(gen_edge_tts(text, edge_voice, fname))
            return True
        except: return False
    return False

def gen_image_hf(prompt, w, h):
    if not hf_token: return None
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}
    full_prompt = prompt + ", high quality illustration, isometric, cinematic lighting, no text"
    full_prompt += ", vertical 9:16" if w < h else ", wide 16:9"
    
    for _ in range(3):
        try:
            res = requests.post(API_URL, headers=headers, json={"inputs": full_prompt}, timeout=20)
            if res.status_code == 200: return res.content
            time.sleep(1)
        except: time.sleep(1)
    return None

def process_scene(args):
    line, w, h, tone = args
    if "|" not in line: return None
    parts = line.split("|")
    if len(parts) < 2: return None
    
    img_p = parts[0].replace("Scene", "").replace(":", "").strip()
    aud_t = parts[1].strip()
    
    # 1. Audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        af = f.name
    if not gen_audio(aud_t, af, tone): return None
    
    # 2. Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        img_data = gen_image_hf(img_p, w, h)
        if img_data:
            f.write(img_data); img_path = f.name
        else:
            img_path = "PLACEHOLDER"
            
    return (af, img_path)

def create_video(script, w, h, tone):
    lines = [l for l in script.split('\n') if "|" in l and "Scene" in l][:10] # Max 10 cảnh
    if not lines: return None
    
    bar = st.progress(0)
    args = [(l, w, h, tone) for l in lines]
    results = []
    
    # Chạy tuần tự để tránh lỗi
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for i, res in enumerate(executor.map(process_scene, args)):
            results.append(res)
            bar.progress((i+1)/len(lines))
            
    clips = []
    for res in results:
        if res:
            af, imgf = res
            try:
                ac = AudioFileClip(af)
                dur = ac.duration + 0.5
                if imgf == "PLACEHOLDER":
                    # Màn hình đen nếu ảnh lỗi
                    clip = ColorClip(size=(w, h), color=(0,0,0), duration=dur)
                else:
                    clip = ImageClip(imgf).set_duration(dur)
                
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

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ Insurance Script")

col1, col2 = st.columns([1, 1.5], gap="medium")

with col1:
    st.subheader("1. Kiến trúc nội dung")
    
    c1, c2 = st.columns(2)
    with c1:
        pillar = st.selectbox("Pillar (Trụ cột)", ["Kiến thức & Giáo dục", "Sản phẩm & Giải pháp", "Niềm tin & Bằng chứng", "Phong cách sống"])
    with c2:
        angle = st.selectbox("Angle (Góc độ)", ["Chuyên gia phân tích", "Kể chuyện (Storytelling)", "Cảnh báo (Drama)", "Hài hước (Fun)", "Q&A Giải đáp"])

    kw = st.text_input("Từ khóa", "Bảo hiểm nhân thọ cho người trụ cột")
    fmt = st.radio("Định dạng", ["Clip (Video)", "Bài Website", "Bài Facebook"])
    
    vw, vh = 1280, 720
    seo_prompt = ""
    
    if fmt == "Clip (Video)":
        ratio = st.radio("Khung hình:", ["Ngang 16:9", "Dọc 9:16"], horizontal=True)
        vw, vh = (1280, 720) if "Ngang" in ratio else (720, 1280)
        dur = st.slider("Giây", 15, 90, 45)
        seo_prompt = f"Viết kịch bản Video {dur}s. Cấu trúc: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'."
    elif fmt == "Bài Website":
        seo_prompt = "Viết bài chuẩn SEO. Dùng thẻ {IMAGE: prompt}."
    else:
        seo_prompt = "Viết Caption Facebook. Đề xuất ảnh vuông."

    # Map Tone
    tone_map = {"Chuyên gia phân tích": "Chuyên nghiệp", "Kể chuyện (Storytelling)": "Cảm động", "Cảnh báo (Drama)": "Chuyên nghiệp", "Hài hước (Fun)": "Hài hước", "Q&A Giải đáp": "Đời thường"}
    auto_tone = tone_map.get(angle, "Chuyên nghiệp")
    st.info(f"🎙️ Tone giọng AI: **{auto_tone}**")

    if st.button("🚀 XỬ LÝ NGAY"):
        if not api_key: st.error("Thiếu Gemini Key")
        elif not hf_token and fmt != "Bài Facebook": st.error("Thiếu HuggingFace Token")
        else:
            with st.spinner("AI đang viết..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(selected_model)
                    prompt = f"""
                    Role: Chuyên gia Content. Topic: {kw}. Pillar: {pillar}. Angle: {angle}.
                    Output:
                    1. Title SEO
                    2. Hashtags
                    3. Content: {seo_prompt}
                    Lưu ý: Không dùng dấu ** trong lời bình.
                    """
                    res = model.generate_content(prompt)
                    st.session_state.res = res.text
                    st.session_state.fmt = fmt
                    st.session_state.sets = {'w': vw, 'h': vh, 'tone': auto_tone}
                except Exception as e: st.error(f"Lỗi: {e}")

with col2:
    st.subheader("2. Kết quả")
    if 'res' in st.session_state:
        r = st.session_state.res
        f = st.session_state.fmt
        
        if f == "Bài Website":
            st.image(gen_image_hf(f"{kw} header", 1200, 628) or "https://via.placeholder.com/800", use_container_width=True)
            st.markdown(r)
        elif f == "Bài Facebook":
            st.image(gen_image_hf(f"{kw} square", 1080, 1080) or "https://via.placeholder.com/800", width=450)
            st.markdown(r)
        else:
            st.caption(f"Tone: {st.session_state.sets['tone']} | Server: {tts_provider}")
            if st.button("🎥 Dựng Video"):
                v = create_video(r, st.session_state.sets['w'], st.session_state.sets['h'], st.session_state.sets['tone'])
                if v: st.video(v)
            st.text_area("Kịch bản", r, height=500)
