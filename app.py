import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
# Import chuẩn cho xử lý video
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- CẤU HÌNH ---
st.set_page_config(page_title="DAT Media AI Workflow", layout="wide", page_icon="🛡️")

# --- CSS LÀM ĐẸP ---
st.markdown("""
    <style>
    .stButton>button {background-color: #FF4B4B; color: white; font-weight: bold; border-radius: 8px;}
    .reportview-container {background: #f0f2f6;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH THÔNG MINH ---
with st.sidebar:
    st.title("⚙️ Cấu hình hệ thống")
    
    # 1. Nhập API Key
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Đã kết nối API từ hệ thống")
    else:
        api_key = st.text_input("Nhập API Key", type="password")
    
    # 2. Tự động quét mô hình (QUAN TRỌNG)
    st.divider()
    st.markdown("### 🧠 Chọn bộ não AI")
    
    available_models = ["models/gemini-pro"] # Mặc định an toàn
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Lấy danh sách model thực tế từ Google
            all_models = genai.list_models()
            # Chỉ lấy các model hỗ trợ tạo nội dung (generateContent)
            available_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        except Exception as e:
            st.error(f"Không lấy được danh sách model: {e}")

    # Cho người dùng chọn model có sẵn, không lo bị sai tên
    selected_model = st.selectbox(
        "Mô hình đang hoạt động:", 
        available_models,
        index=0 if available_models else 0,
        help="Chọn 'gemini-1.5-flash' nếu có để chạy nhanh nhất. Nếu lỗi, chọn 'gemini-pro'."
    )

# --- HÀM XỬ LÝ (BACKEND) ---
def get_image_url(keyword):
    """Lấy ảnh minh họa miễn phí"""
    clean_keyword = keyword.replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{clean_keyword}?width=1280&height=720&nologo=true"

def create_video_from_script(script_data):
    """Dựng video từ kịch bản"""
    clips = []
    try:
        lines = script_data.strip().split('\n')
        for line in lines:
            if "|" in line and "Scene" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    img_prompt = parts[0].replace("Scene", "").replace(":", "").strip()
                    voice_text = parts[1].strip()
                    
                    # 1. Tạo Audio
                    tts = gTTS(text=voice_text, lang='vi')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                        tts.save(audio_file.name)
                        audio_path = audio_file.name
                    
                    # 2. Tải ảnh
                    img_url = get_image_url(img_prompt)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
                        img_file.write(requests.get(img_url).content)
                        img_path = img_file.name
                    
                    # 3. Ghép Clip
                    audio_clip = AudioFileClip(audio_path)
                    clip = ImageClip(img_path).set_duration(audio_clip.duration + 0.5)
                    clip = clip.set_audio(audio_clip)
                    clip = clip.set_fps(24)
                    clips.append(clip)
        
        if clips:
            final_video = concatenate_videoclips(clips, method="compose")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                final_video.write_videofile(temp_video.name, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')
                return temp_video.name
    except Exception as e:
        st.error(f"Lỗi dựng phim: {str(e)}")
        return None

# --- GIAO DIỆN CHÍNH (FRONTEND) ---
st.title("🛡️ AI Content Generator: Bảo Hiểm & Tài Chính")
st.caption(f"Đang sử dụng mô hình: **{selected_model}**")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Đầu vào nội dung")
    keyword = st.text_input("Từ khóa / Chủ đề", "Bảo hiểm nhân thọ trọn đời")
    sector = st.selectbox("Lĩnh vực", ["Bảo hiểm Nhân thọ", "Bảo hiểm Phi nhân thọ", "Chăm sóc sức khỏe", "Tài chính cá nhân"])
    
    # --- CÁC TÙY CHỌN ĐẦY ĐỦ ---
    content_type = st.radio("Định dạng đầu ra", ["Clip (Video)", "Bài Website", "Bài Facebook"])
    
    tone_dict = {
        "Chuyên nghiệp": "Tin cậy, số liệu rõ ràng, nghiêm túc.",
        "Đời thường": "Gần gũi, dùng từ ngữ dân dã, thân thiện.",
        "Hài hước": "Vui vẻ, bắt trend, dùng emoji.",
        "Kể chuyện (Storytelling)": "Dẫn dắt bằng câu chuyện cảm động hoặc tình huống thực tế."
    }
    tone_key = st.selectbox("Tone giọng & Phong cách", list(tone_dict.keys()))
    
    # Tùy biến theo định dạng
    extra_prompt = ""
    if content_type == "Clip (Video)":
        st.info("💡 AI sẽ: Viết kịch bản -> Vẽ ảnh -> Đọc Voice -> Dựng Video")
        duration = st.slider("Thời lượng video (giây)", 30, 90, 40)
        extra_prompt = f"Viết kịch bản Video ngắn {duration} giây. BẮT BUỘC trả về định dạng từng dòng: 'Scene [số]: [Mô tả ảnh tiếng Anh] | [Lời bình tiếng Việt]'"
    elif content_type == "Bài Website":
        words = st.number_input("Số từ", 500, 2000, 800)
        extra_prompt = f"Viết bài chuẩn SEO website {words} từ. Có các thẻ H1, H2, H3. Đề xuất vị trí chèn ảnh."
    else: # Facebook
        extra_prompt = "Viết bài Facebook ngắn gọn, viral, nhiều emoji, tập trung tương tác."

    btn_process = st.button("🚀 BẮT ĐẦU XỬ LÝ")

# --- XỬ LÝ KẾT QUẢ ---
with col2:
    st.subheader("2. Kết quả")
    
    if btn_process:
        if not api_key:
            st.error("Vui lòng nhập API Key trước!")
        else:
            with st.spinner("AI đang suy nghĩ và viết bài..."):
                try:
                    # Gọi Gemini với Model đã chọn từ danh sách thực tế
                    model = genai.GenerativeModel(selected_model)
                    full_prompt = f"""
                    Vai trò: Chuyên gia Content Marketing ngành {sector}.
                    Chủ đề: {keyword}
                    Tone giọng: {tone_key} ({tone_dict[tone_key]})
                    Yêu cầu: {extra_prompt}
                    
                    Lưu ý: Nếu là Video, hãy tuân thủ tuyệt đối định dạng 'Scene X: [Visual Prompt] | [Audio Script]' để máy có thể đọc được.
                    """
                    
                    response = model.generate_content(full_prompt)
                    st.session_state.result_text = response.text
                    st.session_state.content_type = content_type
                    st.success("Đã có nội dung!")
                    
                except Exception as e:
                    st.error(f"Lỗi: {e}. \n\n👉 Hãy thử đổi mô hình khác ở cột bên trái.")

    # Hiển thị kết quả
    if 'result_text' in st.session_state:
        # Nếu là Video -> Tự động dựng phim
        if st.session_state.content_type == "Clip (Video)":
            tab1, tab2 = st.tabs(["🎬 Video Demo", "📝 Kịch bản gốc"])
            
            with tab1:
                if st.button("🎥 Bấm vào đây để Dựng Video"):
                    with st.spinner("Đang vẽ ảnh và ghép giọng đọc (khoảng 1 phút)..."):
                        video_path = create_video_from_script(st.session_state.result_text)
                        if video_path:
                            st.video(video_path)
                            with open(video_path, "rb") as v_file:
                                st.download_button("⬇️ Tải Video về máy", v_file, "video_demo.mp4")
                        else:
                            st.warning("Không dựng được video. Hãy kiểm tra kịch bản bên tab kia xem có đúng định dạng Scene X: ... | ... không.")
            
            with tab2:
                st.text_area("Kịch bản thô", st.session_state.result_text, height=400)
        
        # Nếu là Bài viết -> Hiển thị text
        else:
            st.markdown(st.session_state.result_text)
            st.button("Copy nội dung")
