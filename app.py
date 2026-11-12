import pickle
import numpy as np
import streamlit as st
from Orange.data import Domain, Table

# 페이지 기본 설정
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측",
    page_icon="🚦",
    layout="wide",
)

# 1) 모델 파일 로드 (캐싱 적용으로 빠른 반응 속도 유지)
@st.cache_resource
def load_orange_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


try:
    model = load_orange_model("traffic_casualty_model.pkcls")
except FileNotFoundError:
    st.error("모델 파일(`traffic_casualty_model.pkcls`)을 찾을 수 없습니다. 같은 폴더에 파일을 배치해주세요.")
    st.stop()
except Exception as e:
    st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# 모델의 원본 도메인(피처 정보) 추출
orig_domain = model.original_domain
sido_attr = orig_domain["시도"]
sido_options = list(sido_attr.values)  # ['경기', '부산', '서울']

# 헤더 영역
st.title("🚦 교통사고 예상 사망자 수 예측 및 어린이 보호구역 안전 대시보드")
st.markdown(
    "Orange3로 학습된 신경망(`MLPRegressor`) 모델을 기반으로 **시도**와 **사고·사상자 지표**를 조절하여 "
    "예상 사망자 수와 단계별 안전 수칙을 실시간으로 확인합니다."
)
st.divider()

# 2) 사이드바 / 입력 컨트롤 (시도 선택 및 슬라이더)
with st.sidebar:
    st.header("🎛️ 사고 조건 입력")

    selected_sido = st.selectbox(
        "시도 선택",
        options=sido_options,
        index=sido_options.index("서울") if "서울" in sido_options else 0,
    )

    accident_count = st.slider(
        "사고건수 (건)",
        min_value=0,
        max_value=5000,
        value=500,
        step=10,
    )

    minor_injury = st.slider(
        "경상자수 (명)",
        min_value=0,
        max_value=5000,
        value=350,
        step=10,
    )

    with st.expander("추가 모델 입력 변수 설정 (중상자수·부상신고자수)", expanded=False):
        st.caption(
            "학습된 모델은 5개 변수(시도, 사고건수, 중상자수, 경상자수, 부상신고자수)를 사용합니다. "
            "필요 시 아래 값을 직접 조정할 수 있습니다."
        )
        serious_injury = st.slider(
            "중상자수 (명)",
            min_value=0,
            max_value=2000,
            value=int(accident_count * 0.25),
            step=5,
        )
        reported_injury = st.slider(
            "부상신고자수 (명)",
            min_value=0,
            max_value=1000,
            value=int(accident_count * 0.05),
            step=5,
        )

# Orange3 Table 형태로 입력 데이터 구성 후 실시간 예측 실행
input_domain = Domain(orig_domain.attributes)
sido_index = float(sido_attr.to_val(selected_sido))
input_row = np.array(
    [[sido_index, float(accident_count), float(serious_injury), float(minor_injury), float(reported_injury)]],
    dtype=float,
)
input_table = Table.from_numpy(input_domain, input_row)

raw_pred = float(model(input_table)[0])
predicted_deaths = max(0.0, raw_pred)  # 음수 예측 방지

# 3) 예측 결과에 따른 신호등 단계 판정
if predicted_deaths < 5.0:
    status_key = "SAFE"
    light_emoji = "🟢"
    status_label = "안전 (SAFE)"
    status_color = "#16a34a"
    bg_color = "#dcfce7"
    border_color = "#86efac"
    summary_msg = "예상 사망자 수가 낮은 수준입니다. 기본 보행 안전 수칙을 유지하세요."
    safety_rules = [
        "횡단보도 앞에서는 반드시 한 번 멈추고 좌우를 살핀 뒤 건너기",
        "보행 중 스마트폰 및 이어폰 사용 자제하기",
        "어린이 보호구역 내 제한속도(시속 30km 이하) 준수하기",
        "녹색 신호가 켜져도 차량이 완전히 멈췄는지 확인 후 횡단하기",
    ]
elif predicted_deaths < 15.0:
    status_key = "CAUTION"
    light_emoji = "🟡"
    status_label = "주의 (CAUTION)"
    status_color = "#ca8a04"
    bg_color = "#fef9c3"
    border_color = "#fde047"
    summary_msg = "사고 규모에 따른 인명 피해 가능성이 높아지고 있습니다. 각별한 주의가 필요합니다."
    safety_rules = [
        "어린이 보호구역 내 불법 주·정차 절대 금지 (시야 가림 사고 예방)",
        "골목길·교차로 진입 전 일시정지 및 경적·전조등으로 보행자에게 위치 알리기",
        "어린이는 밝은 색 옷이나 가방 반사 키링을 착용해 시인성 높이기",
        "학원 차량·통학버스 승하차 시 보호자 동행 및 주변 차량 서행 운전",
    ]
else:
    status_key = "DANGER"
    light_emoji = "🔴"
    status_label = "위험 (DANGER)"
    status_color = "#dc2626"
    bg_color = "#fee2e2"
    border_color = "#fca5a5"
    summary_msg = "고위험 구간입니다! 즉각적인 감속 운행과 보행자 집중 보호 조치가 필요합니다."
    safety_rules = [
        "어린이 보호구역 전 구간 시속 20~30km 이하 서행 및 횡단보도 앞 무조건 일시정지",
        "등·하교 시간대 녹색어머니회·교통안전지도사 배치 및 집중 단속 강화",
        "주·정차 차량 사이 무단횡단 원천 차단 및 안전 펜스·과속단속카메라 점검",
        "우회전 시 보행자 신호 유무와 관계없이 일시정지 후 서행 통과",
    ]

# 메인 화면 2단 구성: 좌측(예측 지표 + 대형 신호등) / 우측(어린이 보호구역 안전 수칙 카드)
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📊 실시간 예측 결과")

    m1, m2, m3 = st.columns(3)
    m1.metric("선택 지역", selected_sido)
    m2.metric("입력 사고 / 경상", f"{accident_count:,}건 / {minor_injury:,}명")
    m3.metric("예상 사망자 수", f"{predicted_deaths:.2f} 명")

    # 대형 신호등 UI 카드
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: 3px solid {border_color};
            border-radius: 20px;
            padding: 28px 20px;
            text-align: center;
            margin-top: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        ">
            <div style="
                display: inline-flex;
                gap: 18px;
                background-color: #1f2937;
                padding: 14px 26px;
                border-radius: 50px;
                margin-bottom: 16px;
            ">
                <span style="font-size: 44px; opacity: {'1.0' if status_key == 'SAFE' else '0.22'};">🟢</span>
                <span style="font-size: 44px; opacity: {'1.0' if status_key == 'CAUTION' else '0.22'};">🟡</span>
                <span style="font-size: 44px; opacity: {'1.0' if status_key == 'DANGER' else '0.22'};">🔴</span>
            </div>
            <div style="font-size: 32px; font-weight: 800; color: {status_color}; margin-bottom: 8px;">
                {light_emoji} {status_label}
            </div>
            <div style="font-size: 16px; color: #374151; font-weight: 500;">
                {summary_msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.subheader("🚸 어린이 보호구역(스쿨존) 안전 수칙 카드")

    rules_html = "".join(
        [
            f"""
            <div style="
                background-color: #ffffff;
                border-left: 6px solid {status_color};
                border-top: 1px solid #e5e7eb;
                border-right: 1px solid #e5e7eb;
                border-bottom: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 14px 18px;
                margin-bottom: 12px;
                font-size: 16px;
                color: #111827;
                box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            ">
                <strong>수칙 {idx}.</strong> {rule}
            </div>
            """
            for idx, rule in enumerate(safety_rules, start=1)
        ]
    )

    st.markdown(rules_html, unsafe_allow_html=True)

    st.info(
        f"💡 **신호등 판정 기준 안내**\n"
        f"- 🟢 **안전**: 예상 사망자 수 5명 미만\n"
        f"- 🟡 **주의**: 예상 사망자 수 5명 이상 ~ 15명 미만\n"
        f"- 🔴 **위험**: 예상 사망자 수 15명 이상"
    )
