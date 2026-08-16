import joblib
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from typing import Optional

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None

MODEL_NAME = "gemini-3.6-flash"


def get_streamlit_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def load_api_key() -> Optional[str]:
    load_dotenv()
    return os.getenv("GEMINI_API_KEY") or get_streamlit_secret("GEMINI_API_KEY")


@st.cache_data
def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("./data/student_career_success_dataset.csv", index_col=0)


@st.cache_resource
def create_client(api_key: Optional[str]):
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def build_context(df: pd.DataFrame) -> str:
    def clean_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def add_detail(item_text: str, label: str, value) -> str:
        value = clean_text(value)
        if not value:
            return item_text

        suffix = "" if value.endswith((".", "!", "?")) else "."
        return f"{item_text} {label}: {value}{suffix}"

    menu_lines = []
    for _, row in menu_df.iterrows():
        item_text = f"- {clean_text(row['name'])}: {clean_text(row['description'])}"
        item_text = add_detail(item_text, "Nguyên liệu", row.get("ingredients", ""))
        item_text = add_detail(item_text, "Ghi chú", row.get("notes", ""))
        menu_lines.append(item_text)
    return "\n".join(menu_lines)


st.set_page_config(
    page_title="Sản phẩm cuối khóa - CSI06",
    page_icon="./icons/book-open.svg",
    layout="centered",
)


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("./salary_model.keras")
    scaler = joblib.load("./scaler.pkl")
    model_columns = joblib.load("./model_columns.pkl")
    return model, scaler, model_columns


try:
    model, scaler, model_columns = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Lỗi tải tệp mô hình: {e}")
    st.info(
        'Vui lòng kiểm tra đường dẫn 3 file "salary_model.keras", "scaler.pkl", và "model_columns.pkl".'
    )


GENDER_MAP = {"Nam": "Male", "Nữ": "Female", "Khác": "Other"}

YEAR_MAP = {
    "Năm thứ nhất": "Freshman",
    "Năm thứ hai": "Sophomore",
    "Năm thứ ba": "Junior",
    "Năm thứ tư": "Senior",
}

ACADEMIC_MAP = {
    "Xuất sắc": "Excellent",
    "Tốt": "Good",
    "Trung bình": "Average",
    "Yếu": "Poor",
}

MAJOR_MAP = {
    "Trí tuệ nhân tạo": "Artificial Intelligence",
    "Phân tích kinh doanh": "Business Analytics",
    "Khoa học máy tính": "Computer Science",
    "An ninh mạng": "Cybersecurity",
    "Khoa học dữ liệu": "Data Science",
    "Kỹ thuật điện": "Electrical Engineering",
    "Công nghệ thông tin": "Information Technology",
    "Kỹ thuật phần mềm": "Software Engineering",
}

st.title("Mô hình dự đoán mức lương khởi điểm")
st.markdown("Trần Gia Phát")
st.divider()

with st.sidebar:
    st.logo("./icons/star.svg")
    option = st.sidebar.radio(
        "Chọn nội dung hiển thị", ["Dự đoán mức lương", "Chatbot"]
    )

if assets_loaded:
    if option == "Dự đoán mức lương":
        st.header(
            "Nhập các thông tin học tập và cá nhân của sinh viên bên dưới để hệ thống AI đưa ra mức lương dự đoán."
        )
        st.divider()
        st.subheader("Nhập thông tin của sinh viên")

        col1, col2 = st.columns(2)

        with col1:
            gender_vi = st.selectbox(
                "Giới tính", options=list(GENDER_MAP.keys()), index=0
            )

            university_year_vi = st.select_slider(
                "Năm học", options=list(YEAR_MAP.keys())
            )

            academic_perf_vi = st.pills(
                "Xếp loại học lực", options=list(ACADEMIC_MAP.keys())
            )

            programming = st.slider(
                "Kĩ năng lập trình", min_value=0, max_value=10, value=5, step=1
            )

            communication = st.slider(
                "Kĩ năng giao tiếp", min_value=0, max_value=10, value=5, step=1
            )

            teamwork = st.slider(
                "Kĩ năng làm việc nhóm", min_value=0, max_value=10, value=5, step=1
            )

            problem_solving = st.slider(
                "Kĩ năng giải quyết vấn đề", min_value=0, max_value=10, value=5, step=1
            )

        with col2:
            major_vi = st.selectbox("Ngành học", options=list(MAJOR_MAP.keys()))
            attendance = st.slider(
                "Tỉ lệ có mặt", min_value=0, max_value=100, value=50, step=1
            )

            hours = st.number_input(
                "Số giờ học trong tuần",
                min_value=0,
                value=None,
                step=1,
                placeholder="Nhập số giờ học",
            )

            cgpa = st.slider(
                "Điểm trung bình tích lũy (CGPA)",
                min_value=0.0,
                max_value=4.0,
                value=2.0,
                step=0.01,
            )

            projects_completed = st.number_input(
                "Số dự án hoàn thành",
                min_value=0,
                value=None,
                step=1,
                placeholder="Nhập dự án hoàn thành",
            )

            certifications = st.number_input(
                "Số chứng chỉ có",
                min_value=0,
                value=None,
                step=1,
                placeholder="Nhập số chứng chỉ có",
            )

            internships = st.number_input(
                "Số lần thực tập",
                min_value=0,
                value=None,
                step=1,
                placeholder="Nhập số lần thực tập",
            )

        st.divider()

        if st.button(
            "Dự Đoán Mức Lương Khởi Điểm", type="primary", use_container_width=True
        ):
            input_data = pd.DataFrame(0, index=[0], columns=model_columns)

            mapping_num = {
                "cgpa": cgpa,
                "gpa": cgpa,
                "programming": programming,
                "communication": communication,
                "teamwork": teamwork,
                "problem": problem_solving,
                "attendance": attendance,
                "hours": hours,
                "projects": projects_completed,
                "certifications": certifications,
                "internships": internships,
            }

            for col in input_data.columns:
                for key, val in mapping_num.items():
                    if key in col.lower():
                        input_data[col] = val

            gender_en = GENDER_MAP[gender_vi]
            year_en = YEAR_MAP[university_year_vi]
            academic_en = ACADEMIC_MAP.get(academic_perf_vi, "Good")
            major_en = MAJOR_MAP[major_vi]

            encoded_features = [
                f"Gender_{gender_en}",
                f"University_Year_{year_en}",
                f"Academic_Performance_{academic_en}",
                f"Major_{major_en}",
            ]

            for feat in encoded_features:
                if feat in input_data.columns:
                    input_data[feat] = 1

            input_scaled = scaler.transform(input_data)
            print(input_data)
            pred_log = model.predict(input_scaled)[0][0]
            pred_usd = np.expm1(pred_log)

            st.balloons()
            st.success("### Kết Quả Dự Đoán AI")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Lương ước tính (USD / Năm)", f"${pred_usd:,.2f}")
            st.badge(
                f"Mức lương khởi điểm này được tính thông qua những giá trị đầu vào ở trên bằng một mô hình được huấn luyện."
            )

    elif option == "Chatbot":
        st.header("Trợ Lý AI Tư Vấn Dự Án")
        st.write(
            "Giải đáp các thắc mắc về tập dữ liệu `student_career_success_dataset.csv` và mô hình dự đoán mức lương."
        )
        st.divider()

        api_key = load_api_key()
        client = create_client(api_key)

        if not client:
            st.warning(
                "Chưa cấu hình GEMINI_API_KEY. Vui lòng thêm API Key vào file `.env` hoặc `secrets.toml` để kích hoạt Chatbot."
            )
        else:
            try:
                config_data = load_config()
                start_msg = config_data.get(
                    "initial_bot_message", "Chào bạn! Tôi có thể giúp gì cho bạn?"
                )
            except Exception:
                start_msg = (
                    "Chào bạn! Tôi là SalaryBot tư vấn dự án mức lương sinh viên."
                )

            SYSTEM_INSTRUCTION = """
            Bạn là SalaryBot - trợ lý AI thông minh tư vấn dự án 'Dự đoán mức lương khởi điểm sinh viên' của Trần Gia Phát (Lớp CSI06).

            Thông tin dự án & dữ liệu:
            1. Tập dữ liệu: `student_career_success_dataset.csv` gồm 50,000 bản ghi sinh viên với các biến số: CGPA, Ngành học (Computer Science, Data Science, AI, Cybersecurity, Software Engineering...), Kỹ năng (Programming, Communication, Teamwork, Problem Solving), Số giờ học, Tỷ lệ chuyên cần, Dự án, Chứng chỉ, Thực tập.
            2. Mô hình AI: Mạng Nơ-ron Nhân Tạo (Neural Network / Deep Learning) TensorFlow/Keras.
            3. Tiền xử lý dữ liệu: Sử dụng One-Hot Encoding cho các biến phân loại, StandardScaler để chuẩn hóa dữ liệu đầu vào và chuyển đổi Log (np.log1p) đối với mức lương mục tiêu (Starting_Salary_USD).

            Hãy phản hồi bằng tiếng Việt ngắn gọn, rõ ràng, lịch sự và đúng chuyên môn.
            """

            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": start_msg}
                ]

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if prompt := st.chat_input("Hỏi tôi về tập dữ liệu hoặc mô hình AI..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("AI đang suy nghĩ..."):
                        try:
                            contents = []
                            for m in st.session_state.messages:
                                role = "user" if m["role"] == "user" else "model"
                                contents.append(
                                    types.Content(
                                        role=role,
                                        parts=[types.Part.from_text(text=m["content"])],
                                    )
                                )

                            gen_config = types.GenerateContentConfig(
                                system_instruction=SYSTEM_INSTRUCTION,
                                temperature=0.7,
                            )

                            response = client.models.generate_content(
                                model=MODEL_NAME,
                                contents=contents,
                                config=gen_config,
                            )

                            bot_reply = response.text
                            st.write(bot_reply)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": bot_reply}
                            )
                        except Exception as e:
                            st.error(f"Lỗi khi gửi yêu cầu đến Gemini API: {e}")
