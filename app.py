import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="AI Bảo Hiểm - DAT Media", layout="wide", page_icon="🛡️")

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {width: 100%; border-radius: 8px; height: 3em; background-color: #0068C9; color: white; font-weight: bold;}
    .stSuccess {background-color: #D4EDDA; color: #155724;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH KỸ THUẬT (QUAN TRỌNG) ---
with st.sidebar:
    st.title("⚙️ Trung tâm điều khiển")
    
    # 1. NHẬP API KEY (LINH HOẠT HƠN)
    # Ưu tiên lấy từ Secrets, nếu không có thì hiện ô nhập
    api_key_input = ""
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Đã kết nối API từ hệ thống")
    else:
        api_key = st.text_input("Nhập Gemini API Key", type="password")
        if not api_key:
            st.warning("⚠️ Vui lòng nhập API Key để chạy")
            st.stop()
    
    genai.configure(api_key=api_key)

    st.divider()

    # 2. CHỌN MÔ HÌNH (GIẢI QUYẾT VẤN ĐỀ CỦA BẠN)
    st.markdown("### 🧠 Chọn bộ não AI")
    model_option = st.selectbox(
        "Mô hình xử lý:",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        index=0, # Mặc định chọn Flash
        help="Flash: Nhanh & Ổn định. Pro: Viết hay hơn nhưng có thể bị giới hạn số lần dùng."
    )
    
    # 3. ĐỘ SÁNG TẠO
    creativity = st.slider("Độ sáng tạo (Temperature)", 0.0, 1.0, 0.7, 
                           help="Thấp (0.2): Chính xác, logic. Cao (0.8): Bay bổng, kể chuyện.")

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ AI Workflow: Sáng Tạo Nội Dung Bảo Hiểm")
st.caption(f"Đang sử dụng mô hình: **{model_option}**")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("1. Đầu vào nội dung")
    
    keyword = st.text_input("Chủ đề / Từ khóa", "Bảo hiểm trách nhiệm dân sự xe ô tô")
    sector = st.selectbox("Lĩnh vực", ["Bảo hiểm Nhân thọ", "Bảo hiểm Phi nhân thọ", "Chăm sóc sức khỏe"])
    
    format_type = st.radio("Định dạng", ["Clip (Video Ngắn)", "Bài Website chuẩn SEO", "Bài Facebook Viral"])
    
    # Logic hiển thị cấu hình chi tiết
    if format_type == "Clip (Video Ngắn)":
        st.info("💡 AI sẽ tạo kịch bản phân cảnh chi tiết (Visual + Audio)")
        duration = st.slider("Thời lượng (giây)", 30, 120, 60)
        detail_prompt = f"Kịch bản Video ngắn {duration} giây. Chia cột Visual và Audio rõ ràng."
    elif format_type == "Bài Website chuẩn SEO":
        st.info("💡 AI sẽ viết bài dài, chuẩn SEO, phân bổ từ khóa")
        words = st.number_input("Số từ dự kiến", 500, 3000, 1000)
        detail_prompt = f"Bài viết Website chuẩn SEO, độ dài khoảng {words} từ. Cần có Meta Description và các thẻ H1, H2."
    else:
        st.info("💡 AI sẽ viết Caption thu hút + Ý tưởng ảnh")
        detail_prompt = "Bài đăng Facebook văn phong thu hút, tập trung tương tác, nhiều emoji."

    tone = st.select_slider("Tone giọng", options=["Hài hước", "Đời thường", "Chuyên nghiệp", "Chuyên gia cao cấp"])
    
    # Quản lý Feedback
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    
    with st.expander(f"Lịch sử dạy AI ({len(st.session_state.feedback_history)} ghi nhớ)"):
        st.write(st.session_state.feedback_history)
        if st.button("Xóa bộ nhớ tạm"):
            st.session_state.feedback_history = []
            st.rerun()

    btn_run = st.button("🚀 BẮT ĐẦU XỬ LÝ")

# --- HÀM GỌI GEMINI (CÓ XỬ LÝ LỖI) ---
def call_gemini(prompt_text):
    try:
        # Cấu hình model dựa trên lựa chọn ở Sidebar
        generation_config = {
            "temperature": creativity,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        model = genai.GenerativeModel(
            model_name=model_option,
            generation_config=generation_config
        )
        
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"❌ **LỖI:** {str(e)}\n\n💡 *Gợi ý: Hãy thử đổi sang mô hình khác (ví dụ từ Pro sang Flash) ở thanh bên trái.*"

# --- HIỂN THỊ KẾT QUẢ ---
with col2:
    st.header("2. Kết quả")
    
    if btn_run:
        with st.spinner(f"AI đang viết với mô hình {model_option}..."):
            # Tạo prompt tổng hợp
            feedback_str = "\n".join([f"- {fb}" for fb in st.session_state.feedback_history])
            
            final_prompt = f"""
            Vai trò: Chuyên gia Content Marketing ngành Bảo hiểm ({sector}).
            Nhiệm vụ: {detail_prompt}
            Chủ đề: "{keyword}"
            Tone giọng: {tone}.
            
            YÊU CẦU BẮT BUỘC TỪ NGƯỜI DÙNG (Feedback cũ):
            {feedback_str}
            
            Cấu trúc trả về (Markdown):
            1. Tiêu đề hấp dẫn
            2. Hashtags & Keywords
            3. Nội dung chính (Kịch bản phân cảnh hoặc Bài viết hoàn chỉnh)
            4. Đề xuất hình ảnh (Image Prompts)
            """
            
            result = call_gemini(final_prompt)
            st.session_state.result_cache = result
            st.success("Xong!")

    if 'result_cache' in st.session_state:
        tabs = st.tabs(["📄 Nội dung", "🎧 Voice Demo", "💬 Tinh chỉnh"])
        
        with tabs[0]:
            st.markdown(st.session_state.result_cache)
        
        with tabs[1]:
            # Đọc 200 ký tự đầu tiên
            if st.button("Tạo Voice (Demo)"):
                try:
                    clean_text = st.session_state.result_cache.replace("*", "").replace("#", "")[:300]
                    tts = gTTS(text=clean_text, lang='vi')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                        st.audio(fp.name)
                except Exception as e:
                    st.error(f"Không tạo được voice: {e}")

        with tabs[2]:
            new_fb = st.text_input("Góp ý cho AI (Ví dụ: 'Đừng viết dài dòng', 'Thêm số liệu')")
            if st.button("Lưu góp ý"):
                st.session_state.feedback_history.append(new_fb)
                st.success("Đã học!")
