import streamlit as st
import time
import os
import re
import requests
import asyncio
import tempfile
import concurrent.futures

# --- 1. CẤU HÌNH APP ---
st.set_page_config(page_title="Insurance Script", layout="wide", page_icon="🛡️")

# --- 2. KIỂM TRA THƯ VIỆN & FFMPEG ---
try:
    import google.generativeai as genai
    import edge_tts
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip
    import imageio_ffmpeg
    
    # Kiểm tra FFmpeg có thực sự tồn tại không
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if not os.path.exists(ffmpeg_path):
        st.error("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy FFmpeg. Video sẽ không chạy được.")
        st.info("👉 Vào GitHub tạo file `packages.txt` và viết chữ `ffmpeg` vào đó.")
except ImportError as e:
    st.error(f"❌ Thiếu thư viện: {e}. Hãy kiểm tra file requirements.txt")
    st.stop()

# --- CSS ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; height: 3em; width: 100%; border-radius: 8px;}
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    .reportview-container {background: #f0f2f6;}
    </style>
""", unsafe_allow_html=True)

# --- QUẢN LÝ TRẠNG THÁI ---
if 'history' not in st.session_state: st.session_state.history = []
if 'generated_content' not in st.session_state: st.session_state.generated_content = None

# --- CẤU HÌNH VOICE ID ---
VOICE_MAP = {
    "Chuyên nghiệp": "mJLZ5p8I7Pk81BHpKwbx",  # Nam Sadoma
    "Đời thường": "foH7s9fX31wFFH2yqrFa",     # Huyen
    "Cảm động": "1l0C0QA9c9jN22EmWiB0",       # Jade
    "Hài hước": "JxmKvRaNYFidf0N27Vng"        # Son Tran
}

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("🎛️ Cấu hình Hệ thống")
    if st.button("🔄 LÀM MỚI APP"):
        st.session_state.clear()
        st.rerun()

    # Nhập Key
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    eleven_api = st.secrets.get("ELEVEN_API_KEY", "").strip()
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "").strip()

    if api_key: st.success("✅ Gemini: OK")
    else: st.error("❌ Thiếu Gemini Key")
    
    if eleven_api: st.success("✅ ElevenLabs: OK")
    if hf_token: st.success("✅ HuggingFace: OK")

    st.divider()

    # Chọn Model (An toàn: Chỉ cho chọn model có thật)
    st.subheader("🧠 Bộ não xử lý")
    model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    selected_model = st.selectbox("Chọn Model:", model_options)

    # Chọn Giọng
    st.subheader("🔊 Giọng đọc Video")
    tts_provider = st.selectbox("Server:", ["ElevenLabs (VIP)", "Microsoft (Miễn phí)"])
    edge_voice = "vi-VN-HoaiMyNeural"
    if "Microsoft" in tts_provider:
        edge_voice = st.selectbox("Giọng MS:", ["vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"]).split(" ")[0]

# --- CORE FUNCTIONS ---

def clean_text(text):
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    for p in ["Lời bình:", "Audio:", "Voice:", "Scene \d+:", "MC:", "Host:"]:
        text = re.sub(f'{p}', '', text, flags=re.IGNORECASE)
    return text.replace("*", "").replace("#", "").strip()

async def gen_edge_tts(text, voice, fname):
    await edge_tts.Communicate(text, voice).save(fname)

def gen_audio(text, fname, tone):
    text = clean_text(text)
    if not text: return False
    
    # ElevenLabs Logic
    if "ElevenLabs" in tts_provider:
        if not eleven_api: return False
        vid = VOICE_MAP.get(tone, "mJLZ5p8I7Pk81BHpKwbx")
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
            headers = {"xi-api-key": eleven_api, "Content-Type": "application/json"}
            # Dùng model turbo v2.5 mới nhất cho nhanh
            data = {"text": text, "model_id": "eleven_turbo_v2_5"}
            res = requests.post(url, json=data, headers=headers, timeout=60)
            if res.status_code == 200:
                with open(fname, 'wb') as f: f.write(res.content)
                return True
        except: pass

    # Microsoft Logic (Fallback hoặc Chính)
    try:
        asyncio.run(gen_edge_tts(text, edge_voice, fname))
        return True
    except: return False

def gen_image(prompt, w, h):
    """Chỉ dùng HuggingFace để tránh Rate Limit"""
    if not hf_token: return None
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    full_prompt = prompt + ", high quality illustration, isometric, no text"
    full_prompt += ", vertical 9:16" if w < h else ", wide 16:9"
    
    for _ in range(3): # Thử 3 lần
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: af = f.name
    if not gen_audio(aud_t, af, tone): return None
    
    # 2. Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        img_data = gen_image(img_p, w, h)
        if img_data: f.write(img_data); img_path = f.name
        else: img_path = "PLACEHOLDER"
            
    return (af, img_path)

def create_video(script, w, h, tone):
    # Lọc lấy các dòng Scene
    lines = [l for l in script.split('\n') if "|" in l and ("Scene" in l or "Cảnh" in l)][:12]
    if not lines:
        st.error("⚠️ Lỗi kịch bản: Không tìm thấy dòng 'Scene X: ... | ...'")
        return None
        
    st.info(f"🎬 Đang dựng {len(lines)} cảnh...")
    bar = st.progress(0)
    
    # Xử lý tuần tự (Sequential) để ổn định nhất
    clips = []
    for i, line in enumerate(lines):
        res = process_scene((line, w, h, tone))
        if res:
            af, imgf = res
            try:
                ac = AudioFileClip(af)
                dur = ac.duration + 0.5
                
                if imgf == "PLACEHOLDER":
                    clip = ColorClip(size=(w, h), color=(0,0,0), duration=dur)
                else:
                    clip = ImageClip(imgf).set_duration(dur)
                
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

# --- UI CHÍNH: INSURANCE SCRIPT ---
st.title("🛡️ Insurance Script: Content Architect")

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("1. Thiết kế Nội dung")
    
    # INPUT PILLAR & ANGLE
    c1, c2 = st.columns(2)
    with c1:
        pillar = st.selectbox("Pillar (Trụ cột)", ["Kiến thức & Giáo dục", "Sản phẩm & Giải pháp", "Niềm tin & Bằng chứng", "Phong cách sống"])
    with c2:
        angle = st.selectbox("Angle (Góc độ)", ["Chuyên gia phân tích", "Kể chuyện (Storytelling)", "Cảnh báo (Drama)", "Hài hước (Fun)", "Q&A Giải đáp"])

    # INPUT CHI TIẾT
    kw = st.text_input("Chủ đề / Từ khóa", "Bảo hiểm nhân thọ trọn đời")
    
    # CHỌN ĐỊNH DẠNG (Khôi phục đầy đủ)
    st.write("---")
    st.write("📦 **Định dạng đầu ra:**")
    fmt = st.radio("Chọn loại nội dung:", ["Clip (Video)", "Bài Website", "Bài Facebook"], horizontal=True, label_visibility="collapsed")
    
    # CẤU HÌNH CHI TIẾT THEO ĐỊNH DẠNG
    video_w, video_h = 1280, 720
    seo_guide = ""
    
    if fmt == "Clip (Video)":
        st.caption("Cấu hình Video:")
        vc1, vc2 = st.columns(2)
        with vc1:
            ratio = st.radio("Khung hình:", ["Ngang 16:9", "Dọc 9:16"])
            video_w, video_h = (1280, 720) if "Ngang" in ratio else (720, 1280)
        with vc2:
            dur = st.slider("Thời lượng (s)", 15, 90, 45)
        
        # Prompt ép buộc định dạng Video
        seo_guide = f"""
        - Viết kịch bản Video {dur} giây.
        - BẮT BUỘC mỗi cảnh phải viết đúng định dạng: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'
        - Không được dùng dấu ** hay in đậm trong phần lời bình.
        """
        
    elif fmt == "Bài Website":
        words = st.slider("Số từ", 500, 2000, 1000)
        seo_guide = f"- Viết bài chuẩn SEO {words} từ. BẮT BUỘC chèn thẻ {{IMAGE: prompt tiếng Anh}} xen kẽ vào bài."
        
    else: # Facebook
        seo_guide = "- Viết Caption Facebook thu hút, viral. Đề xuất ý tưởng ảnh vuông."

    # TỰ ĐỘNG CHỌN TONE
    tone_map = {"Chuyên gia phân tích": "Chuyên nghiệp", "Kể chuyện (Storytelling)": "Cảm động", "Cảnh báo (Drama)": "Chuyên nghiệp", "Hài hước (Fun)": "Hài hước", "Q&A Giải đáp": "Đời thường"}
    auto_tone = tone_map.get(angle, "Chuyên nghiệp")
    st.info(f"🎙️ Tone giọng AI: **{auto_tone}** (Dựa theo Angle)")

    if st.button("🚀 XỬ LÝ NGAY"):
        if not api_key: st.error("❌ Thiếu Gemini API Key")
        else:
            with st.spinner("AI đang viết nội dung..."):
                try:
                    # GỌI GEMINI (CÓ TRY-EXCEPT ĐỂ KHÔNG SẬP APP)
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(selected_model)
                    
                    prompt = f"""
                    Vai trò: Chuyên gia Content Bảo Hiểm.
                    Topic: {kw}. Pillar: {pillar}. Angle: {angle}.
                    
                    YÊU CẦU ĐẦU RA:
                    1. Tiêu đề (Title)
                    2. 5 Hashtags
                    3. Nội dung chính:
                    {seo_guide}
                    """
                    
                    response = model.generate_content(prompt)
                    
                    # Lưu kết quả vào Session
                    st.session_state.generated_content = response.text
                    st.session_state.fmt = fmt
                    st.session_state.settings = {'w': video_w, 'h': video_h, 'tone': auto_tone}
                    st.success("Đã xong! Xem kết quả bên phải 👉")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối AI: {e}")
                    st.warning("Gợi ý: Hãy thử đổi Model khác trong Sidebar hoặc kiểm tra lại Key.")

with col2:
    st.subheader("2. Kết quả hiển thị")
    
    if st.session_state.generated_content:
        res = st.session_state.generated_content
        ft = st.session_state.fmt
        sets = st.session_state.settings
        
        # HIỂN THỊ THEO ĐỊNH DẠNG
        if ft == "Bài Website":
            st.info("🖼️ Ảnh Featured (HuggingFace)")
            img = gen_image(f"{kw} insurance header", 1200, 628)
            if img: st.image(img, use_container_width=True)
            st.markdown(res)
            
        elif ft == "Bài Facebook":
            st.info("📱 Ảnh Vuông (HuggingFace)")
            img = gen_image(f"{kw} flat lay", 1080, 1080)
            if img: st.image(img, width=450)
            st.markdown(res)
            
        else: # VIDEO MODE
            tab1, tab2 = st.tabs(["🎥 Video Demo", "📝 Kịch bản Chi tiết"])
            
            with tab1:
                st.caption(f"Tone: {sets['tone']} | Server: {tts_provider}")
                if st.button("🎬 BẤM ĐỂ DỰNG VIDEO"):
                    video_file = create_video(res, sets['w'], sets['h'], sets['tone'])
                    if video_file:
                        st.video(video_file)
                        with open(video_file, "rb") as f:
                            st.download_button("⬇️ Tải Video", f, "demo_video.mp4")
            
            with tab2:
                st.text_area("Kịch bản thô", res, height=600)
