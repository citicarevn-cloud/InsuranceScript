import streamlit as st
import time
import os
import re
import requests
import asyncio
import tempfile
import edge_tts
import imageio_ffmpeg
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip, TextClip, CompositeVideoClip
import google.generativeai as genai

# --- 1. CẤU HÌNH APP ---
st.set_page_config(page_title="Insurance Script Pro", layout="wide", page_icon="🛡️")

# --- CSS ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%;}
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    .caption {font-style: italic; color: #555; text-align: center; font-size: 0.9em;}
    </style>
""", unsafe_allow_html=True)

# --- KIỂM TRA FFMPEG ---
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
if not os.path.exists(ffmpeg_path):
    st.error("❌ LỖI: Không tìm thấy FFmpeg. Hãy tạo file `packages.txt` trên GitHub với nội dung `ffmpeg`.")

# --- QUẢN LÝ SESSION ---
if 'history' not in st.session_state: st.session_state.history = []
if 'video_settings' not in st.session_state: st.session_state.video_settings = {'w': 1280, 'h': 720}

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

    # 1. NHẬP KEY
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    eleven_api = st.secrets.get("ELEVEN_API_KEY", "").strip()
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "").strip()

    if not api_key: api_key = st.text_input("Gemini API Key", type="password")
    if not eleven_api: eleven_api = st.text_input("ElevenLabs API Key", type="password")
    if not hf_token: hf_token = st.text_input("HuggingFace Token", type="password")

    if api_key: st.success("✅ Gemini: OK")
    if eleven_api: st.success("✅ ElevenLabs: OK")
    if hf_token: st.success("✅ HuggingFace: OK")

    st.divider()
    
    # 2. MODULE QUÉT MODEL
    st.subheader("🧠 Bộ não xử lý")
    available_models = []
    if api_key:
        try:
            genai.configure(api_key=api_key)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except: pass
    
    if not available_models:
        available_models = ["models/gemini-1.5-flash", "models/gemini-pro"]
        
    selected_model = st.selectbox("Chọn Model:", available_models, index=0)

    st.divider()

    # 3. GIỌNG ĐỌC
    st.subheader("🔊 Giọng đọc Video")
    tts_provider = st.selectbox("Server:", ["ElevenLabs (VIP)", "Microsoft (Miễn phí)"])
    edge_voice = "vi-VN-HoaiMyNeural"
    if "Microsoft" in tts_provider:
        edge_voice = st.selectbox("Giọng MS:", ["vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"]).split(" ")[0]

# --- HÀM XỬ LÝ ---

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
            st.warning("⚠️ Thiếu ElevenLabs Key. Dùng Microsoft thay thế.")
        else:
            vid = VOICE_MAP.get(tone, "mJLZ5p8I7Pk81BHpKwbx")
            try:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
                headers = {"xi-api-key": eleven_api, "Content-Type": "application/json"}
                data = {"text": text, "model_id": "eleven_turbo_v2_5"}
                res = requests.post(url, json=data, headers=headers, timeout=60)
                if res.status_code == 200:
                    with open(fname, 'wb') as f: f.write(res.content)
                    return True
            except: pass

    try:
        asyncio.run(gen_edge_tts(text, edge_voice, fname))
        return True
    except: return False

def gen_image_safe(prompt, w, h):
    """Chiến thuật Hybrid: HF -> Pollinations -> Stock -> Placeholder"""
    # 1. Hugging Face
    if hf_token:
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_token}"}
        full_prompt = prompt + ", masterpiece, high quality, corporate insurance style, no text"
        full_prompt += ", vertical 9:16 portrait" if w < h else ", wide 16:9 landscape"
        try:
            res = requests.post(API_URL, headers=headers, json={"inputs": full_prompt}, timeout=15)
            if res.status_code == 200: 
                time.sleep(2) # Delay nhẹ
                return res.content
        except: pass

    # 2. Pollinations (Thêm delay và User-Agent để tránh Rate Limit)
    try:
        clean_prompt = prompt.replace(" ", "%20")
        url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width={w}&height={h}&nologo=true&seed={int(time.time())}&model=flux"
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Thử tối đa 2 lần
        for _ in range(2):
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 200: return res.content
            time.sleep(2)
    except: pass

    # 3. Stock Backup (Picsum - Không bao giờ lỗi)
    try:
        stock_url = f"https://picsum.photos/seed/{int(time.time())}/{w}/{h}"
        res = requests.get(stock_url, timeout=10)
        if res.status_code == 200: return res.content
    except: pass

    return None

# --- HÀM XỬ LÝ BÀI VIẾT WEBSITE (CAPTION TIẾNG VIỆT) ---
def render_mixed_content(text):
    """
    Xử lý thẻ {IMAGE: English Prompt | Caption Tiếng Việt}
    """
    # Regex tìm thẻ {IMAGE: ...}
    parts = re.split(r'\{IMAGE:\s*(.*?)\}', text, flags=re.IGNORECASE)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            # Xử lý phần trong ngoặc
            raw_content = part.strip()
            
            # Tách Prompt (Anh) và Caption (Việt)
            if "|" in raw_content:
                prompt_en, caption_vn = raw_content.split("|", 1)
            else:
                prompt_en = raw_content
                caption_vn = "Hình ảnh minh họa" # Fallback nếu AI quên tạo caption

            prompt_en = prompt_en.strip()
            caption_vn = caption_vn.strip()

            if prompt_en:
                with st.spinner(f"🎨 Đang vẽ: {caption_vn}..."):
                    # Gọi hàm Safe (Có delay + fallback)
                    img_data = gen_image_safe(prompt_en, 800, 450)
                    if img_data:
                        st.image(img_data, use_container_width=True)
                        st.markdown(f"<div class='caption'>{caption_vn}</div>", unsafe_allow_html=True)
                        time.sleep(2) # Nghỉ 2s sau mỗi ảnh để tránh Rate Limit
                    else:
                        st.warning(f"⚠️ Không tải được ảnh: {caption_vn}")

def create_video(script, w, h, tone):
    lines = [l for l in script.split('\n') if "|" in l and ("Scene" in l or "Cảnh" in l)][:10]
    if not lines:
        st.error("⚠️ Lỗi kịch bản: Không tìm thấy dòng 'Scene X: ... | ...'")
        return None
        
    st.info(f"🎬 Đang xử lý {len(lines)} cảnh...")
    bar = st.progress(0)
    
    clips = []
    
    for i, line in enumerate(lines):
        parts = line.split("|")
        if len(parts) < 2: continue
        
        img_p = parts[0].replace("Scene", "").replace(":", "").strip()
        aud_t = parts[1].strip()
        
        # 1. Audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: af = f.name
        if not gen_audio(aud_t, af, tone): continue
        
        # 2. Image
        img_data = gen_image_safe(img_p, w, h)
        if img_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(img_data); img_path = f.name
        else:
            img_path = "PLACEHOLDER"
        
        # 3. Clip
        try:
            ac = AudioFileClip(af)
            dur = ac.duration + 0.5
            
            if img_path == "PLACEHOLDER":
                txt_clip = TextClip("Đang tải ảnh...", fontsize=30, color='white', size=(w,h)).set_duration(dur)
                bg_clip = ColorClip(size=(w, h), color=(0,50,100), duration=dur)
                clip = CompositeVideoClip([bg_clip, txt_clip])
            else:
                clip = ImageClip(img_path).set_duration(dur)
            
            clip = clip.set_audio(ac).set_fps(15)
            clips.append(clip)
        except: pass
        
        bar.progress((i+1)/len(lines))
        
    if clips:
        try:
            final = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                final.write_videofile(f.name, codec='libx264', audio_codec='aac', fps=15, preset='ultrafast', threads=4)
            bar.empty()
            return f.name
        except Exception as e:
            st.error(f"❌ Lỗi Render: {e}")
            return None
    return None

# --- UI CHÍNH ---
st.title("🛡️ Insurance Script Pro")

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("1. Thiết kế Nội dung")
    
    c1, c2 = st.columns(2)
    with c1:
        pillar = st.selectbox("Pillar (Trụ cột)", ["Kiến thức & Giáo dục", "Sản phẩm & Giải pháp", "Niềm tin & Bằng chứng", "Phong cách sống"])
    with c2:
        angle = st.selectbox("Angle (Góc độ)", ["Chuyên gia phân tích", "Kể chuyện (Storytelling)", "Cảnh báo (Drama)", "Hài hước (Fun)", "Q&A Giải đáp"])

    kw = st.text_input("Chủ đề / Từ khóa", "Bảo hiểm nhân thọ trọn đời")
    
    st.write("---")
    fmt = st.radio("Định dạng:", ["Clip (Video)", "Bài Website", "Bài Facebook"], horizontal=True)
    
    vw, vh = 1280, 720
    seo_guide = ""
    
    if fmt == "Clip (Video)":
        ratio = st.radio("Khung hình:", ["Ngang 16:9", "Dọc 9:16"])
        vw, vh = (1280, 720) if "Ngang" in ratio else (720, 1280)
        dur = st.slider("Thời lượng (s)", 15, 90, 45)
        seo_guide = f"Viết kịch bản Video {dur}s. BẮT BUỘC mỗi dòng: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'"
        
    elif fmt == "Bài Website":
        words = st.slider("Số từ", 500, 2000, 1000)
        # SỬA PROMPT ĐỂ TẠO CAPTION TIẾNG VIỆT
        seo_guide = f"""
        Viết bài chuẩn SEO {words} từ. 
        QUY TẮC CHÈN ẢNH (BẮT BUỘC):
        Mỗi đoạn văn hãy chèn một thẻ ảnh theo định dạng sau:
        {{IMAGE: [Mô tả ảnh chi tiết bằng tiếng Anh để vẽ] | [Caption ngắn gọn tiếng Việt dưới 7 từ]}}
        Ví dụ: {{IMAGE: A happy family holding hands in a park | Gia đình hạnh phúc bên nhau}}
        """
        
    else:
        seo_guide = "Viết Caption Facebook thu hút. Đề xuất ảnh vuông."

    tone_map = {"Chuyên gia phân tích": "Chuyên nghiệp", "Kể chuyện (Storytelling)": "Cảm động", "Cảnh báo (Drama)": "Chuyên nghiệp", "Hài hước (Fun)": "Hài hước", "Q&A Giải đáp": "Đời thường"}
    auto_tone = tone_map.get(angle, "Chuyên nghiệp")
    st.info(f"🎙️ Tone tự động: **{auto_tone}**")

    if st.button("🚀 XỬ LÝ NGAY"):
        if not api_key: st.error("❌ Thiếu Gemini Key")
        else:
            with st.spinner("AI đang viết..."):
                try:
                    model = genai.GenerativeModel(selected_model)
                    
                    prompt = f"""
                    Vai trò: Chuyên gia Content Bảo Hiểm.
                    Topic: {kw}. Pillar: {pillar}. Angle: {angle}.
                    YÊU CẦU:
                    1. Tiêu đề
                    2. Hashtags
                    3. Nội dung: {seo_guide}
                    Lưu ý: Không dùng dấu ** trong lời bình video.
                    """
                    response = model.generate_content(prompt)
                    st.session_state.res = response.text
                    st.session_state.fmt = fmt
                    st.session_state.sets = {'w': vw, 'h': vh, 'tone': auto_tone}
                    st.session_state.kw = kw 
                    st.success("Đã xong!")
                except Exception as e:
                    st.error(f"❌ Lỗi AI: {e}")

with col2:
    st.subheader("2. Kết quả")
    if 'res' in st.session_state:
        res = st.session_state.res
        ft = st.session_state.fmt
        sets = st.session_state.sets
        kw_saved = st.session_state.get('kw', 'insurance')
        
        if ft == "Bài Website":
            # Ảnh Featured
            st.info("🖼️ Ảnh Featured")
            feat_img = gen_image_safe(f"{kw_saved} insurance header illustration", 1200, 628)
            if feat_img: 
                st.image(feat_img, use_container_width=True)
                st.markdown("<div class='caption'>Ảnh đại diện bài viết</div>", unsafe_allow_html=True)
            
            st.write("---")
            render_mixed_content(res)
            
        elif ft == "Bài Facebook":
            st.info("📱 Ảnh Vuông")
            img = gen_image_safe(f"{kw_saved} flat lay", 1080, 1080)
            if img: st.image(img, width=450)
            st.markdown(res)
            
        else: # Video
            tab1, tab2 = st.tabs(["🎥 Video", "📝 Kịch bản"])
            with tab1:
                st.caption(f"Server: {tts_provider} | Tone: {sets['tone']}")
                if st.button("🎬 BẤM ĐỂ DỰNG VIDEO"):
                    video_file = create_video(res, sets['w'], sets['h'], sets['tone'])
                    if video_file:
                        st.video(video_file)
            with tab2:
                st.text_area("Script", res, height=600)
