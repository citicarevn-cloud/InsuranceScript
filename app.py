import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
import re # Thư viện quan trọng để tách ảnh trong bài viết
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Content Bảo Hiểm", layout="wide", page_icon="🛡️")

# --- CSS LÀM ĐẸP ---
st.markdown("""
    <style>
    .stButton>button {background-color: #0068C9; color: white; font-weight: bold; border-radius: 8px; height: 3em;}
    img {border-radius: 8px; margin-top: 15px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .caption {font-style: italic; color: #555; text-align: center; font-size: 0.9rem;}
    h2 {color: #0068C9;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API đã kết nối")
    else:
        api_key = st.text_input("Nhập API Key", type="password")
    
    st.divider()
    available_models = ["models/gemini-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            all_models = genai.list_models()
            available_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        except: pass
    # Ưu tiên chọn Flash hoặc Pro 1.5 nếu có
    default_index = 0
    for i, m in enumerate(available_models):
        if "1.5" in m: default_index = i; break
        
    selected_model = st.selectbox("Model:", available_models, index=default_index)

# --- HÀM XỬ LÝ CORE (ĐÃ NÂNG CẤP) ---

def get_image_url(prompt, width=1280, height=720):
    """
    Tạo URL ảnh. Đã thêm bộ lọc để HẠN CHẾ ẢNH CHÂN DUNG NGƯỜI THẬT.
    Chuyển sang phong cách minh họa (illustration), conceptual để an toàn và chuyên nghiệp hơn.
    """
    # Thêm các từ khóa định hướng phong cách để tránh ảnh người thật cận mặt
    style_modifiers = ", conceptual illustration, isometric style, flat design, business concept, no photorealistic portraits"
    full_prompt = prompt + style_modifiers
    
    clean_prompt = full_prompt.replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&nologo=true&seed={os.urandom(4)}"

def render_mixed_content(text):
    """Hàm biên tập: Tách văn bản và ảnh từ thẻ {{IMAGE:...}}"""
    # Regex tìm chuỗi nằm giữa {{IMAGE: và }}
    parts = re.split(r'\{\{IMAGE:(.*?)\}\}', text, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Phần văn bản
            if part.strip(): st.markdown(part)
        else:
            # Phần prompt ảnh (nằm trong thẻ)
            img_prompt = part.strip()
            with st.spinner(f"🤖 Đang vẽ minh họa: {img_prompt[:30]}..."):
                # Tạo ảnh với size chữ nhật nằm ngang cho bài viết
                img_url = get_image_url(img_prompt, width=800, height=450)
                st.image(img_url, use_container_width=True)

def create_video_from_script(script_data):
    """Dựng video (Giữ nguyên logic cũ)"""
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
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as af:
                        tts.save(af.name); audio_path = af.name
                    
                    # Ảnh video cũng áp dụng bộ lọc no-portrait
                    img_url = get_image_url(img_prompt)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as imgf:
                        imgf.write(requests.get(img_url).content); img_path = imgf.name
                    
                    ac = AudioFileClip(audio_path)
                    clip = ImageClip(img_path).set_duration(ac.duration+0.5).set_audio(ac).set_fps(24)
                    clips.append(clip)
        
        if clips:
            final_video = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
                final_video.write_videofile(tf.name, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')
                return tf.name
    except Exception as e:
        st.error(f"Lỗi dựng phim: {str(e)}"); return None

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ AI Content Generator: Bảo Hiểm & Tài Chính")

col1, col2 = st.columns([1, 1.5], gap="medium")

with col1:
    st.subheader("1. Nhập liệu")
    keyword = st.text_input("Chủ đề / Từ khóa", "Bảo hiểm nhân thọ cho người trụ cột")
    sector = st.selectbox("Lĩnh vực", ["Bảo hiểm Nhân thọ", "Bảo hiểm Phi nhân thọ", "Chăm sóc sức khỏe", "Tài chính cá nhân"])
    
    content_type = st.radio("Định dạng", ["Bài Website chuẩn SEO", "Bài Facebook Viral", "Clip (Video Ngắn)"])
    
    tone_options = ["Chuyên gia tin cậy, khách quan", "Người đồng hành, thấu cảm", "Kể chuyện đời thường, gần gũi"]
    tone_key = st.selectbox("Tone giọng", tone_options)
    
    # --- PROMPT NÂNG CẤP (MẠNH MẼ HƠN) ---
    extra_prompt = ""
    if content_type == "Clip (Video Ngắn)":
        duration = st.slider("Thời lượng (s)", 30, 90, 45)
        extra_prompt = f"Viết kịch bản Video {duration}s. BẮT BUỘC trả về định dạng từng dòng: 'Scene X: [Mô tả ảnh tiếng Anh, tập trung vào đồ vật/bối cảnh] | [Lời bình tiếng Việt]'"
        
    elif content_type == "Bài Website chuẩn SEO":
        words = st.number_input("Số từ tối thiểu", 600, 2500, 1000)
        # Prompt cực mạnh để ép AI chèn thẻ ảnh
        extra_prompt = f"""
        Viết bài chuẩn SEO {words} từ. Sử dụng các thẻ H2, H3 để chia đoạn.
        YÊU CẦU CẤU TRÚC BẮT BUỘC (RẤT QUAN TRỌNG):
        1. Bài viết phải có ít nhất 2-3 hình ảnh minh họa xen kẽ trong phần nội dung chính.
        2. Tại vị trí muốn chèn ảnh, bạn phải viết CHÍNH XÁC dòng code này: {{IMAGE: mô tả cảnh vật, concept, đồ vật bằng tiếng Anh (tránh mô tả người cụ thể)}}.
        3. AI vẽ ảnh sẽ đọc lệnh trong {{IMAGE:...}} để tạo hình.
        4. Ví dụ: 
           ...nội dung đoạn 1...
           {{IMAGE: illustration of a financial safety net concept}}
           Chú thích: Bảo hiểm là lưới an toàn tài chính.
           ## H2 Tiêu đề tiếp theo...
        """
        
    else: # Facebook
        extra_prompt = "Viết caption Facebook thu hút, tập trung vào nỗi đau hoặc lợi ích khách hàng, dùng emoji phù hợp. Gợi ý 1 ý tưởng ảnh vuông ở cuối bài."

    btn_process = st.button("🚀 TẠO NỘI DUNG", type="primary")

# --- XỬ LÝ KẾT QUẢ ---
with col2:
    st.subheader("2. Kết quả")
    
    if btn_process:
        if not api_key: st.error("Thiếu API Key")
        else:
            with st.spinner("AI đang phân tích và sáng tạo..."):
                try:
                    model = genai.GenerativeModel(selected_model)
                    # Thêm yêu cầu tránh mô tả người trong prompt chính
                    full_prompt = f"""
                    Vai trò: Chuyên gia Content Marketing ngành {sector}. 
                    Chủ đề: {keyword}. Tone giọng: {tone_key}.
                    Lưu ý chung: Khi mô tả hình ảnh, hãy tập trung vào các khái niệm (concept), đồ vật, bối cảnh, tránh mô tả chân dung người cụ thể.
                    {extra_prompt}
                    """
                    response = model.generate_content(full_prompt)
                    st.session_state.final_result = response.text
                    st.session_state.final_type = content_type
                    st.session_state.final_keyword = keyword
                    st.success("Đã xong! Đang tải hình ảnh...")
                except Exception as e:
                    st.error(f"Lỗi AI: {e}. Hãy thử đổi Model khác.")

    if 'final_result' in st.session_state:
        # A. WEBSITE
        if st.session_state.final_type == "Bài Website chuẩn SEO":
            st.markdown(f"### 🖼️ Ảnh Featured: {st.session_state.final_keyword}")
            # Ảnh Featured (1200x628), thêm từ khóa concept để tránh người
            feat_prompt = f"{st.session_state.final_keyword} insurance concept, header banner, wide angle"
            st.image(get_image_url(feat_prompt, 1200, 628), use_container_width=True)
            
            st.divider()
            st.markdown("### 📄 Nội dung bài viết")
            # Gọi hàm render thông minh để hiển thị bài viết + ảnh trong bài
            render_mixed_content(st.session_state.final_result)
            
        # B. FACEBOOK
        elif st.session_state.final_type == "Bài Facebook Viral":
            st.markdown("### 📱 Ảnh đại diện Facebook (Vuông)")
            # Ảnh vuông (1080x1080)
            fb_prompt = f"{st.session_state.final_keyword} insurance concept, creative flat lay composition, instagram style"
            st.image(get_image_url(fb_prompt, 1080, 1080), width=450)
            st.divider()
            st.markdown("### 💬 Caption")
            st.write(st.session_state.final_result)

        # C. VIDEO
        else:
            tab1, tab2 = st.tabs(["🎬 Xem Video", "📝 Kịch bản thô"])
            with tab1:
                if st.button("🎥 Dựng Video ngay (Mất ~1 phút)"):
                    with st.spinner("Đang vẽ ảnh và ghép voice..."):
                        v_path = create_video_from_script(st.session_state.final_result)
                        if v_path: st.video(v_path)
            with tab2:
                st.text_area("Raw Script", st.session_state.final_result, height=400)
