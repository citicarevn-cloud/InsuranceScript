import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import requests
from moviepy.editor import ImageClip, AudioFileClip, ConcatenateAudioClip, CompositeVideoClip, TextClip

# --- CẤU HÌNH ---
st.set_page_config(page_title="AI Bảo Hiểm - Video Generator", layout="wide", page_icon="🎬")

# --- HÀM HỖ TRỢ: TÌM ẢNH MIỄN PHÍ ---
# Dùng dịch vụ source.unsplash.com (đã đóng) thay bằng pollinations (AI Image Generator miễn phí cực nhanh)
def get_image_url(keyword):
    # Tạo ảnh minh họa bằng AI miễn phí qua URL
    clean_keyword = keyword.replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{clean_keyword}?width=1280&height=720&nologo=true"

# --- HÀM HỖ TRỢ: TẠO VIDEO ---
def create_video_segment(text, image_prompt, duration=5):
    # 1. Tạo Audio từ Text
    tts = gTTS(text=text, lang='vi')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
        tts.save(audio_file.name)
        audio_path = audio_file.name

    # 2. Tải ảnh về
    img_url = get_image_url(image_prompt)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
        img_data = requests.get(img_url).content
        img_file.write(img_data)
        img_path = img_file.name

    # 3. Dựng Clip bằng MoviePy
    audio_clip = AudioFileClip(audio_path)
    # Ảnh hiện lâu bằng độ dài audio + 0.5s nghỉ
    clip_duration = audio_clip.duration + 0.5
    
    video_clip = ImageClip(img_path).set_duration(clip_duration)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip = video_clip.set_fps(24) # FPS thấp cho nhẹ
    
    return video_clip

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.success("✅ API Connected")
    else:
        api = st.text_input("Nhập API Key")
        if api: genai.configure(api_key=api)

st.title("🎬 AI Tạo Video Demo Bảo Hiểm")
st.caption("Lưu ý: Video này là bản NHÁP (Draft) để duyệt nội dung. Dùng CapCut để làm đẹp sau.")

col1, col2 = st.columns([1, 1.5])

with col1:
    topic = st.text_input("Chủ đề", "Bảo hiểm du lịch quốc tế")
    tone = st.selectbox("Giọng đọc", ["Nữ nhẹ nhàng", "Nam trầm ấm"]) # gTTS chỉ có 1 giọng, đây là giả lập logic
    
    if st.button("🎥 LÊN KỊCH BẢN & DỰNG VIDEO"):
        st.session_state.processing = True

# --- XỬ LÝ LOGIC ---
if st.session_state.get('processing'):
    with col2:
        # BƯỚC 1: VIẾT KỊCH BẢN
        with st.status("1. AI đang viết kịch bản...", expanded=True) as status:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Viết kịch bản video ngắn (khoảng 30-40 giây) về: {topic}.
            Chia làm đúng 3 phân cảnh (Scene).
            Trả về định dạng thuần:
            Scene 1: [Mô tả hình ảnh tiếng Anh ngắn gọn] | [Lời bình tiếng Việt]
            Scene 2: [Mô tả hình ảnh tiếng Anh ngắn gọn] | [Lời bình tiếng Việt]
            Scene 3: [Mô tả hình ảnh tiếng Anh ngắn gọn] | [Lời bình tiếng Việt]
            Không thêm gì khác.
            """
            response = model.generate_content(prompt)
            script_content = response.text
            st.text_area("Kịch bản thô", script_content, height=150)
            status.update(label="✅ Đã xong kịch bản!", state="complete", expanded=False)

        # BƯỚC 2: DỰNG VIDEO (RENDER)
        with st.status("2. Đang vẽ ảnh & Dựng video (Mất khoảng 1 phút)...", expanded=True) as status:
            try:
                # Phân tích kịch bản
                lines = script_content.strip().split('\n')
                clips = []
                
                for line in lines:
                    if "|" in line:
                        parts = line.split("|")
                        img_prompt = parts[0].replace("Scene", "").strip() # Lấy prompt vẽ ảnh
                        voice_text = parts[1].strip() # Lấy lời bình
                        
                        st.write(f"🎞️ Đang xử lý: {img_prompt}...")
                        clip = create_video_segment(voice_text, img_prompt)
                        clips.append(clip)
                
                # Ghép các đoạn lại
                if clips:
                    final_video = concatenate_videoclips(clips, method="compose")
                    
                    # Xuất file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                        final_video.write_videofile(temp_video.name, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')
                        st.session_state.video_path = temp_video.name
                    
                    status.update(label="✅ Đã dựng xong Video!", state="complete")
                else:
                    st.error("Không đọc được kịch bản. Thử lại nhé.")
                    
            except Exception as e:
                st.error(f"Lỗi dựng phim: {str(e)}")
                # Cần import thêm ở đầu file nếu lỗi: from moviepy.editor import concatenate_videoclips

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state.get('video_path'):
    with col2:
        st.success("🎉 VIDEO CỦA BẠN ĐÃ SẴN SÀNG!")
        st.video(st.session_state.video_path)
        
        # Nút tải xuống
        with open(st.session_state.video_path, "rb") as file:
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=file,
                file_name="baohiem_demo.mp4",
                mime="video/mp4"
            )
            
        st.info("💡 Mẹo: Hình ảnh trong video được AI vẽ tự động. Bạn có thể mang kịch bản này sang CapCut để thay bằng video thật.")

# --- SỬA LỖI IMPORT THIẾU ---
from moviepy.editor import concatenate_videoclips
