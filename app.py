import pickle
import numpy as np
import streamlit as st
from Orange.data import Table, Domain

# 페이지 기본 설정
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측",
    page_icon="🚦",
    layout="centered"
)

# 모델 로드 (캐싱을 통해 속도 최적화)
@st.cache_resource
def load_orange_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

st.title("🚦 어린이 보호구역 교통사고 위험도 예측기")
st.markdown("오렌지3(Orange3) 인공신경망 회귀 모델을 활용하여 사고 규모에 따른 **예상 사망자 수**와 **위험 단계**를 실시간으로 분석합니다.")

# 1) 모델 파일 로드
MODEL_PATH = "traffic_casualty_model.pkcls"
try:
    model = load_orange_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"모델 파일(`{MODEL_PATH}`)을 찾을 수 없습니다. 깃허브 저장소에 파일이 업로드되었는지 확인해주세요.")
    st.stop()
except Exception as e:
    st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

st.divider()

# 2) 입력 UI 구성 (시도 선택 및 슬라이더)
st.subheader("📊 사고 발생 조건 입력")

# 모델의 도메인에서 '시도' 카테고리 목록 추출 ('경기', '부산', '서울')
sido_var = model.domain["시도"]
sido_options = list(sido_var.values)

col1, col2 = st.columns(2)

with col1:
    selected_sido = st.selectbox(
        "📍 시도 선택",
        options=sido_options,
        index=sido_options.index("서울") if "서울" in sido_options else 0
    )

with col2:
    accidents = st.slider(
        "🚗 사고건수 (건)",
        min_value=0,
        max_value=200,
        value=25,
        step=1
    )

minor_injuries = st.slider(
    "🩹 경상자수 (명)",
    min_value=0,
    max_value=200,
    value=15,
    step=1
)

# 모델 학습 시 포함된 나머지 2개 피처(중상자수, 부상신고자수)는 사이드바에서 선택적 조절 가능
with st.sidebar:
    st.header("⚙️ 세부 변수 설정")
    st.caption("모델 학습에 함께 사용된 추가 변수입니다. 기본값은 학습 데이터의 평균치로 설정되어 있습니다.")
    serious_injuries = st.slider("🏥 중상자수 (명)", min_value=0, max_value=100, value=12, step=1)
    reported_injuries = st.slider("📋 부상신고자수 (명)", min_value=0, max_value=50, value=2, step=1)

# 실시간 예측 실행
# Orange3 모델 도메인 순서: ['시도', '사고건수', '중상자수', '경상자수', '부상신고자수']
sido_idx = sido_var.to_val(selected_sido)
input_row = [
    sido_idx,
    float(accidents),
    float(serious_injuries),
    float(minor_injuries),
    float(reported_injuries)
]

# Orange Table 형태로 변환하여 모델 예측 호출 (전처리 Normalizer 자동 적용)
input_domain = Domain(model.domain.attributes)
input_table = Table.from_numpy(input_domain, np.array([input_row], dtype=float))

raw_prediction = model(input_table)[0]
predicted_deaths = max(0.0, float(raw_prediction))  # 음수 예측 방지

st.divider()

# 3) 예측 결과 및 대형 신호등 / 안전 수칙 카드 출력
st.subheader("🚨 실시간 예측 및 위험도 진단")

# 위험도 기준 설정 (필요에 따라 임계값 조정 가능)
if predicted_deaths < 0.3:
    status_emoji = "🟢"
    status_title = "안전 (SAFE)"
    bg_color = "#E8F5E9"
    border_color = "#2E7D32"
    text_color = "#1B5E20"
    guideline_title = "🟢 [안전 단계] 어린이 보호구역 기본 안전 수칙"
    guidelines = [
        "**스쿨존 30km/h 이하 서행**: 시야가 확보되더라도 언제든 어린이가 뛰어나올 수 있음을 유의하세요.",
        "**횡단보도 앞 일시정지**: 보행자가 없더라도 횡단보도 앞에서는 반드시 일시정지 후 출발하세요.",
        "**불법 주·정차 절대 금지**: 주·정차된 차량은 운전자의 시야를 가리는 가장 큰 사고 원인입니다."
    ]
elif predicted_deaths < 1.0:
    status_emoji = "🟡"
    status_title = "주의 (CAUTION)"
    bg_color = "#FFFDE7"
    border_color = "#F9A825"
    text_color = "#F57F17"
    guideline_title = "🟡 [주의 단계] 사고 다발 구간 집중 방어운전 수칙"
    guidelines = [
        "**등·하교 시간대 우회 및 감속**: 어린이 통행이 집중되는 시간대에는 시속 20km 이하로 서행하세요.",
        "**교차로 우회전 시 횡단보도 확인**: 우회전 직후 만나는 횡단보도에서 보행자 신호를 반드시 확인하세요.",
        "**전방 주시 및 스마트폰 사용 금지**: 경상 사고 빈도가 높은 구간이므로 전방·측면 사각지대를 수시로 살피세요."
    ]
else:
    status_emoji = "🔴"
    status_title = "위험 (DANGER)"
    bg_color = "#FFEBEE"
    border_color = "#C62828"
    text_color = "#B71C1C"
    guideline_title = "🔴 [위험 단계] 고위험 스쿨존 특별 안전 수칙 및 조치"
    guidelines = [
        "**즉시 감속 및 비상등 활용**: 중대 사고 위험이 매우 높은 조건입니다. 주변 차량에 주의를 알리며 서행하세요.",
        "**신호등 없는 횡단보도 완전 정지**: 보행자 유무와 관계없이 3초 이상 완전 정지 후 좌우를 확인하세요.",
        "**안전 시설물 점검 권고**: 과속방지턱, 무인단속카메라, 방호울타리 등 교통안전 시설 보강이 시급한 수준입니다."
    ]

# 대형 신호등 및 예측 수치 카드 렌더링
st.markdown(
    f"""
    <div style="
        background-color: {bg_color};
        border: 3px solid {border_color};
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    ">
        <div style="font-size: 84px; line-height: 1.1; margin-bottom: 10px;">
            {status_emoji}
        </div>
        <div style="font-size: 28px; font-weight: 800; color: {text_color}; margin-bottom: 8px;">
            {status_title}
        </div>
        <div style="font-size: 18px; color: #333333;">
            예상 사망자 수: <strong style="font-size: 26px; color: {text_color};">{predicted_deaths:.2f} 명</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 어린이 보호구역 안전 수칙 카드
with st.container(border=True):
    st.markdown(f"#### {guideline_title}")
    for rule in guidelines:
        st.markdown(f"- {rule}")