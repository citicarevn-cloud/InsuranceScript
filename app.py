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
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media AI Studio", layout="wide", page_icon="🎙️")

# --- CSS ---
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
    api_key_raw = st.secrets.get("GEMINI_API_KEY", "")
    eleven_api_raw = st.secrets.get("ELEVEN_API_KEY", "")
    hf_token_raw = st.secrets.get("HUGGINGFACE_TOKEN", "")

    if not api_key_raw: api_key_raw = st.text_input("Gemini API Key", type="password")
    if not eleven_api_raw: eleven_api_raw = st.text_input("ElevenLabs API Key", type="password")
    if not hf_token_raw: hf_token_raw = st.text_input("HuggingFace Token (BẮT BUỘC)", type="password")
    
    # Clean Keys
    api_key = api_key_raw.strip() if api_key_raw else ""
    eleven_api = eleven_api_raw.strip() if eleven_api_raw else ""
    hf_token = hf_token_raw.strip() if hf_token_raw else ""

    if api_key: st.success("✅ Gemini: OK")
    if eleven_api: st.success("✅ ElevenLabs: OK")
    if hf_token: st.success("✅ HuggingFace: OK")
    else: st.error("❌ Thiếu Token HuggingFace (Sẽ không tạo được ảnh)")

    st.divider()
    
    # 2. MODEL GEMINI
    st.subheader("🧠 Bộ não xử lý")
    available_models = ["models/gemini-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except: pass
    selected_model = st.selectbox("Chọn Model:", available_models, index=0)

    st.divider()

    # 3. GIỌNG ĐỌC
    st.subheader("🔊 Nguồn giọng đọc")
    # Đã bỏ tùy chọn Google, chỉ còn nguồn xịn
    tts_provider = st.selectbox("Chọn Server:", ["ElevenLabs (VIP)", "Microsoft (Miễn phí)"])
    
    edge_voice = "vi-VN-HoaiMyNeural" 
    if "Microsoft" in tts_provider:
        edge_voice = st.selectbox("Chọn giọng MS:", [
            "vi-VN-HoaiMyNeural (Nữ)", "vi-VN-NamMinhNeural (Nam)"
        ]).split(" ")[0]

# --- HÀM XỬ LÝ (CORE) ---

def clean_text_for_audio(text):
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

def generate_audio_strict(text, filename, tone_key="Chuyên nghiệp"):
    """
    Hàm tạo Audio KHÔNG CÓ GOOGLE FALLBACK.
    Nếu lỗi là dừng luôn.
    """
    clean_text = clean_text_for_audio(text)
    if not clean_text: return False
    
    # 1. ELEVENLABS STRICT MODE
    if "ElevenLabs" in tts_provider:
        if not eleven_api:
            st.error("❌ Lỗi: Bạn chọn ElevenLabs nhưng chưa nhập API Key!")
            return False
            
        voice_id = VOICE_MAP.get(tone_key, "mJLZ5p8I7Pk81BHpKwbx").strip()
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": eleven_api, "Content-Type": "application/json"}
        # Thử dùng turbo_v2 cho nhanh và ổn định
        data = {"text": clean_text, "model_id": "eleven_turbo_v2"} 
        
        try:
            # Tăng timeout lên 60s
            response = requests.post(url, json=data, headers=headers, timeout=60)
            if response.status_code == 200:
                with open(filename, 'wb') as f: f.write(response.content)
                return True
            else:
                # In thẳng lỗi ra màn hình cho người dùng thấy
                error_msg = response.text
                st.error(f"❌ ElevenLabs từ chối phục vụ: {response.status_code}")
                st.code(error_msg) # Hiện chi tiết lỗi json
                return False # Dừng, không chuyển google
        except Exception as e:
            st.error(f"❌ Lỗi kết nối mạng tới ElevenLabs: {e}")
            return False

    # 2. MICROSOFT STRICT MODE
    if "Microsoft" in tts_provider:
        try:
            asyncio.run(generate_edge_tts(clean_text, edge_voice, filename))
            return True
        except Exception as e:
            st.error(f"❌ Lỗi Microsoft TTS: {e}")
            return False

    return False

def generate_image_huggingface_only(prompt, width, height):
    """
    CHỈ DÙNG HUGGING FACE.
    Không Pollinations -> Không Rate Limit.
    """
    if not hf_token:
        st.error("❌ Chưa có Token Hugging Face. Vui lòng nhập vào Sidebar.")
        return None

    # Sử dụng model SDXL Lightning (Siêu nhanh) hoặc Base
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    style = ", high quality illustration, isometric style, flat design, cinematic lighting, no text"
    full_prompt = prompt + style
    if width < height: full_prompt += ", vertical, 9:16 portrait"
    else: full_prompt += ", wide angle, 16:9 landscape"

    # Thử 3 lần (Retry) nếu HF server bận
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": full_prompt}, timeout=20)
            if response.status_code == 200:
                return response.content
            else:
                # Nếu lỗi 503 (Model đang load) thì đợi chút rồi thử lại
                err_info = response.json()
                if 'estimated_time' in err_info:
                    wait_time = err_info['estimated_time']
                    st.toast(f"⏳ Model đang khởi động, đợi {wait_time:.1f}s...")
                    time.sleep(wait_time + 1)
                else:
                    st.error(f"❌ HF Error {response.status_code}: {response.text}")
                    return None
        except Exception as e:
            st.error(f"❌ Lỗi kết nối HF: {e}")
            time.sleep(1)
            
    return None

def process_scene_strict(args):
    """Xử lý cảnh với quy tắc nghiêm ngặt"""
    part, width, height, tone = args
    try:
        if "|" in part:
            data = part.split("|")
            if len(data) < 2: return None
            
            img_prompt = data[0].replace("Scene", "").replace(":", "").strip()
            raw_voice_text = data[1].strip()
            
            # 1. Tạo Audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                audio_path = f.name
            
            # Gọi hàm Strict (Không Google)
            success = generate_audio_strict(raw_voice_text, audio_path, tone)
            
            if not success:
                st.warning(f"⚠️ Bỏ qua cảnh này do lỗi âm thanh: '{raw_voice_text[:20]}...'")
                return None # Bỏ qua cảnh này luôn

            # 2. Tạo Ảnh (Chỉ HF)
            img_content = generate_image_huggingface_only(img_prompt, width, height)
            
            if img_content:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(img_content); img_path = f.name
                return (audio_path, img_path)
            else:
                # Nếu ảnh lỗi, trả về placeholder đen để giữ tiếng
                return (audio_path, "PLACEHOLDER")
    except Exception as e:
        st.error(f"Lỗi xử lý cảnh: {e}")
        return None

def create_video_strict(script_data, width, height, tone):
    lines = [line for line in script_data.strip().split('\n') if "|" in line and "Scene" in line]
    if len(lines) > 10: lines = lines[:10] # Max 10 cảnh demo
    
    total_scenes = len(lines)
    if total_scenes == 0: return None

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"🚀 Đang xử lý tuần tự (HuggingFace Only + No Google)...")
    
    process_args = [(line, width, height, tone) for line in lines]
    
    results = []
    # Chạy tuần tự để dễ debug lỗi
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for i, result in enumerate(executor.map(process_scene_strict, process_args)):
            results.append(result)
            progress_bar.progress(int((i + 1) / total_scenes * 100))
            
    status_text.text("🎬 Đang render video...")
    clips = []
    for asset in results:
        if asset:
            audio_path, img_path = asset
            try:
                ac = AudioFileClip(audio_path)
                if img_path == "PLACEHOLDER":
                    clip = ColorClip(size=(width, height), color=(0,0,0), duration=ac.duration + 0.5)
                else:
                    clip = ImageClip(img_path).set_duration(ac.duration + 0.5)
                clip = clip.set_audio(ac).set_fps(15)
                clips.append(clip)
            except: pass

    if clips:
        try:
            final = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                final.write_videofile(f.name, codec='libx264', audio_codec='aac', fps=15, preset='ultrafast', threads=4)
            status_text.text("✅ Xong!")
            progress_bar.empty()
            return f.name
        except Exception as e:
            st.error(f"Lỗi Render Video: {e}")
            return None
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
                img_content = generate_image_huggingface_only(img_prompt, width, height)
                if img_content:
                    st.image(img_content, caption=f"🎨 {img_prompt}", use_container_width=True)
                else:
                    st.warning("⚠️ Lỗi tải ảnh (HuggingFace)")

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
        # CHECK NGHIÊM NGẶT CÁC TOKEN
        error = False
        if not api_key: 
            st.error("Chưa nhập Gemini API Key"); error=True
        if content_type != "Bài Facebook" and not hf_token: # Facebook/Web/Video cần ảnh
             st.error("Chưa nhập HuggingFace Token (Không thể tạo ảnh)"); error=True
        if "ElevenLabs" in tts_provider and not eleven_api:
             st.error("Chưa nhập ElevenLabs Key"); error=True

        if not error:
            with st.spinner(f"AI đang tư duy..."):
                try:
                    st.session_state.video_settings = {'w': video_w, 'h': video_h}
                    st.session_state.tone_key = tone
                    
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(selected_model) 
                    
                    past_fb = "\n".join([f"- {fb}" for fb in st.session_state.feedback_history])
                    prompt = f"""
                    Chủ đề: {keyword}. Lĩnh vực: {sector}. Tone: {tone}.
                    YÊU CẦU:
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
                except Exception as e: st.error(f"Lỗi Gemini: {e}")

    if 'result' in st.session_state:
        if st.session_state.type == "Bài Website":
            st.info("🖼️ Ảnh Featured (HuggingFace Only)")
            img_content = generate_image_huggingface_only(f"{st.session_state.kw} insurance header", 1200, 628)
            if img_content: st.image(img_content, use_container_width=True)
            render_mixed_content(st.session_state.result)
        elif st.session_state.type == "Bài Facebook":
            st.info("📱 Ảnh Vuông (HuggingFace Only)")
            img_content = generate_image_huggingface_only(f"{st.session_state.kw} flat lay", 1080, 1080)
            if img_content: st.image(img_content, width=450)
            st.markdown(st.session_state.result)
        else:
            tab1, tab2 = st.tabs(["🎬 Video Demo", "📝 Kịch bản"])
            with tab1:
                vw = st.session_state.video_settings['w']
                vh = st.session_state.video_settings['h']
                tk = st.session_state.get('tone_key', "Chuyên nghiệp")
                
                if "ElevenLabs" in tts_provider:
                    current_id = VOICE_MAP.get(tk, "").strip()
                    st.info(f"🎙️ ElevenLabs: `{current_id}` (No Google Fallback)")
                
                if st.button("🎥 Dựng Video"):
                    # Gọi hàm Strict
                    v_path = create_video_strict(st.session_state.result, vw, vh, tk)
                    if v_path: st.video(v_path)
            with tab2:
                st.text_area("Script", st.session_state.result, height=500)
