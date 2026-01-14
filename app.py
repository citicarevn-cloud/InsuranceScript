import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
import re
import time # Thư viện để xử lý delay tránh rate limit
import random # Thư viện random seed
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media AI Workflow", layout="wide", page_icon="🛡️")

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%;}
    img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0;}
    .reportview-container {background: #f0f2f6;}
    .feedback-box {border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #fff;}
    </style>
""", unsafe_allow_html=True)

# --- QUẢN LÝ SESSION & FEEDBACK ---
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = [] # Lưu lịch sử dạy AI

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    
    # 1. NÚT RESET (Yêu cầu số 2)
    if st.button("🔄 LÀM MỚI (RESET)"):
        # Giữ lại feedback history, chỉ xóa kết quả hiện tại
        saved_history = st.session_state.feedback_history
        st.session_state.clear()
        st.session_state.feedback_history = saved_history
        st.rerun()

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
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except: pass
        
    selected_model = st.selectbox("Chọn Model:", available_models, index=0)
    
    # Hiển thị Feedback đang nhớ
    with st.expander(f"🧠 AI đang nhớ {len(st.session_state.feedback_history)} bài học"):
        for i, fb in enumerate(st.session_state.feedback_history):
            st.text(f"#{i+1}: {fb}")
        if st.button("Xóa trí nhớ"):
            st.session_state.feedback_history = []
            st.rerun()

# --- HÀM XỬ LÝ (BACKEND) ---

def get_image_url(prompt, width=1280, height=720):
    """
    Tạo URL ảnh với cơ chế chống Rate Limit (Yêu cầu số 1)
    """
    # 1. Thêm delay nhẹ 0.5s để server không chặn
    time.sleep(0.5) 
    
    # 2. Random Seed cực mạnh để tránh trùng lặp cache
    seed = random.randint(1, 99999999)
    
    # 3. Prompt style an toàn
    style = ", high quality illustration, isometric style, flat design, vector art, cinematic lighting, no text"
    clean_prompt = (prompt + style).replace(" ", "%20")
    
    return f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&nologo=true&seed={seed}"

def render_mixed_content(text):
    """Hiển thị văn bản & ảnh xen kẽ"""
    pattern = r'\{{1,2}IMAGE:?\s*(.*?)\}{1,2}'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip(): st.markdown(part)
        else:
            img_prompt = part.strip().replace("}", "").replace("{", "")
            with st.container():
                st.write("")
                try:
                    img_url = get_image_url(img_prompt, width=800, height=450)
                    st.image(img_url, caption=f"🎨 Minh họa: {img_prompt}", use_container_width=True)
                except:
                    st.error("Không tải được ảnh do đường truyền kém.")
                st.write("")

def create_video_from_script(script_data):
    """Dựng video"""
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
    keyword = st.text_input("Chủ đề chính", "Bảo hiểm nhân thọ trọn đời")
    sector = st.selectbox("Lĩnh vực", ["Nhân thọ", "Phi nhân thọ", "Sức khỏe", "Tài chính"])
    
    content_type = st.radio("Định dạng", ["Bài Website chuẩn SEO", "Bài Facebook Viral", "Clip (Video)"])
    
    tone = st.select_slider("Tone giọng", ["Hài hước", "Đời thường", "Chuyên nghiệp", "Cảm động"])
    
    # --- LOGIC TẠO PROMPT ---
    seo_guide = ""
    if content_type == "Clip (Video)":
        target_platform = "YouTube/TikTok"
        duration = st.slider("Giây", 30, 90, 45)
        seo_guide = f"""
        - Viết kịch bản Video {duration}s. Cấu trúc mỗi dòng: 'Scene X: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'.
        - Tiêu đề phải giật tít (Clickbait) phù hợp TikTok/Shorts.
        """
    elif content_type == "Bài Website chuẩn SEO":
        target_platform = "Google Search"
        words = st.number_input("Số từ", 500, 2000, 800)
        seo_guide = f"""
        - Viết bài chuẩn SEO {words} từ. Dùng thẻ H2, H3.
        - BẮT BUỘC chèn thẻ {{IMAGE: english prompt}} xen kẽ vào bài.
        - Tiêu đề phải chứa từ khóa chính, tối ưu SEO Google.
        """
    else:
        target_platform = "Facebook Fanpage"
        seo_guide = "- Viết caption thu hút, nhiều emoji. Tiêu đề kích thích tương tác."

    btn_run = st.button("🚀 XỬ LÝ NGAY")

# --- KẾT QUẢ ---
with col2:
    st.subheader("2. Kết quả hiển thị")
    
    if btn_run:
        if not api_key: st.error("Chưa có API Key!")
        else:
            with st.spinner("Đang phân tích từ khóa và viết bài..."):
                try:
                    model = genai.GenerativeModel(selected_model)
                    
                    # Lấy lại các bài học cũ
                    past_lessons = "\n".join([f"- {fb}" for fb in st.session_state.feedback_history])
                    
                    # PROMPT TỔNG HỢP (Yêu cầu 3, 4, 5)
                    final_prompt = f"""
                    Vai trò: Chuyên gia Content SEO ngành {sector}.
                    Nhiệm vụ: Tạo nội dung cho nền tảng {target_platform}.
                    Chủ đề: {keyword}. Tone giọng: {tone}.
                    
                    YÊU CẦU CẤU TRÚC TRẢ VỀ (BẮT BUỘC):
                    1. TIÊU ĐỀ CHUẨN SEO: (Viết 1 tiêu đề thật hay)
                    2. DANH SÁCH KEYWORDS: (5 hashtags #... và 5 tags SEO phù hợp với {target_platform})
                    3. NỘI DUNG CHÍNH:
                       {seo_guide}
                    
                    HÃY ÁP DỤNG CÁC BÀI HỌC TỪ QUÁ KHỨ CỦA NGƯỜI DÙNG:
                    {past_lessons}
                    """
                    
                    response = model.generate_content(final_prompt)
                    st.session_state.result = response.text
                    st.session_state.type = content_type
                    st.session_state.kw = keyword
                    # Xóa trạng thái feedback cũ để nhập mới
                    if 'rating' in st.session_state: del st.session_state.rating
                    if 'comment' in st.session_state: del st.session_state.comment
                    
                    st.success("Xong!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # --- KHU VỰC HIỂN THỊ ---
    if 'result' in st.session_state:
        # A. Xử lý Website
        if st.session_state.type == "Bài Website chuẩn SEO":
            st.info("🖼️ Ảnh Featured")
            # Rate limit fix: Delay 1 chút
            time.sleep(1) 
            feat_prompt = f"{st.session_state.kw} insurance concept header"
            st.image(get_image_url(feat_prompt, 1200, 628), use_container_width=True)
            
            st.markdown("---")
            render_mixed_content(st.session_state.result)
            
        # B. Xử lý Facebook
        elif st.session_state.type == "Bài Facebook Viral":
            st.info("📱 Ảnh Facebook")
            time.sleep(1)
            fb_prompt = f"{st.session_state.kw} insurance creative flat lay"
            st.image(get_image_url(fb_prompt, 1080, 1080), width=450)
            st.markdown(st.session_state.result)
            
        # C. Xử lý Video
        else:
            tab1, tab2 = st.tabs(["🎬 Video Demo", "📝 Kịch bản SEO"])
            with tab1:
                if st.button("🎥 Dựng Video"):
                    with st.spinner("Đang render..."):
                        v = create_video_from_script(st.session_state.result)
                        if v: st.video(v)
            with tab2:
                st.text_area("Script", st.session_state.result, height=400)

        # --- KHU VỰC ĐÁNH GIÁ & HỌC HỎI (Yêu cầu 4) ---
        st.markdown("---")
        st.subheader("⭐ Đánh giá & Dạy AI")
        with st.form("feedback_form"):
            col_f1, col_f2 = st.columns([1, 3])
            with col_f1:
                rating = st.slider("Chất lượng:", 1, 5, 5)
            with col_f2:
                comment = st.text_input("Góp ý cụ thể (AI sẽ ghi nhớ để sửa lần sau):", 
                                      placeholder="Ví dụ: Ảnh cần sáng hơn, giọng văn cần nghiêm túc hơn...")
            
            submitted = st.form_submit_button("Gửi đánh giá")
            if submitted:
                # Logic lưu bài học
                if comment:
                    note = f"Đánh giá {rating} sao. Yêu cầu user: {comment}"
                    st.session_state.feedback_history.append(note)
                    st.success("Đã ghi nhớ! Lần chạy tới AI sẽ áp dụng góp ý này.")
                else:
                    st.success("Cảm ơn bạn đã đánh giá!")
