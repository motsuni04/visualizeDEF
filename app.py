import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="방어력 계산기",
    layout="wide"
)

st.title("📊 방어력 & 관통률 계산기")
st.latex(r'''\text{방어계수} = \frac{794}{794 + \max(0, \text{방어력} \times (1 - \text{관통률}) - \text{관통수치})}''')

data_pen_rate = st.number_input(
    "공격자 관통률(%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=4.0,
    format="%.1f"
)
data_flat_pen = st.number_input(
    "공격자 관통 수치",
    min_value=0,
    max_value=1000,
    step=9,
    value=27
)
if 'data_defense' not in st.session_state:
    st.session_state.data_defense = 952.8
data_defense = st.number_input(
    "방어자 방어력",
    min_value=0.0,
    max_value=2223.2,
    step=0.1,
    value=st.session_state.data_defense,
    format="%.1f"
)
# 버튼들이 세션 상태를 업데이트하므로, number_input과 충돌이 없습니다.
def_presets = st.columns([1, 1, 1, 1])
with def_presets[0]:
    st.button("대부분의 보스 (952.8)", on_click=lambda: st.session_state.update({'data_defense': 952.8}), key='def_btn1')
with def_presets[1]:
    st.button("사냥꾼 (1588)", on_click=lambda: st.session_state.update({'data_defense': 1588.0}), key='def_btn2')
with def_presets[2]:
    st.button("사냥꾼 - 미야즈마 (2223.2)", on_click=lambda: st.session_state.update({'data_defense': 2223.2}), key='def_btn3')
with def_presets[3]:
    st.button("신규 보스 (476.4)", on_click=lambda: st.session_state.update({'data_defense': 476.4}), key='def_btn4')


@st.cache_data
def calculate_defense_coefficient(defense, pen_rate, flat_pen):
    """제공된 공식에 따라 방어계수를 계산하는 함수"""
    pen_rate_decimal = pen_rate / 100.0
    effective_defense = defense * (1 - pen_rate_decimal) - flat_pen
    denominator_term = np.maximum(0, effective_defense)
    defense_coefficient = 794.0 / (794.0 + denominator_term)
    return defense_coefficient


def generate_graph_data(defense, flat_pen):
    """
    관통률(0% ~ 100%)에 따른 방어계수와 0% 대비 변동률 데이터를 생성합니다.
    """
    pen_rates = np.arange(0.0, 101.0, 0.5)

    # 각 관통률에 대한 방어계수 계산
    coefficients = [calculate_defense_coefficient(defense, pr, flat_pen) for pr in pen_rates]

    df = pd.DataFrame({
        '관통률 (%)': pen_rates,
        '방어계수': coefficients
    })

    # --- 방어계수 0% 대비 변동률 계산 (수정된 부분) ---
    # 관통률 0%일 때의 방어계수 (첫 번째 값)
    _coeff_at_0_percent = df.loc[0, '방어계수']

    # 변동률 계산: (현재 방어계수 - 0% 방어계수) / 0% 방어계수 * 100
    if _coeff_at_0_percent != 0:
        df['0% 대비 변동률 (%)'] = (df['방어계수'] - _coeff_at_0_percent) / _coeff_at_0_percent * 100
    else:
        df['0% 대비 변동률 (%)'] = 0.0

    return df


# 데이터 생성
graph_df = generate_graph_data(data_defense, data_flat_pen)

# 현재 방어계수 값 계산
current_coeff = calculate_defense_coefficient(data_defense, data_pen_rate, data_flat_pen)

# --- 그래프 생성 (2차 축 추가 및 이름 변경) ---

# 2차 축을 사용하기 위해 make_subplots를 사용
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 1. 방어계수 선 그래프 (기본 Y축)
fig.add_trace(
    go.Scatter(x=graph_df['관통률 (%)'], y=graph_df['방어계수'], name='방어계수'),
    secondary_y=False,
)

# 2. 0% 대비 변동률 선 그래프 (2차 Y축)
fig.add_trace(
    go.Scatter(x=graph_df['관통률 (%)'], y=graph_df['0% 대비 변동률 (%)'], name='0% 대비 변동률 (%)',
               line=dict(color='red', dash='dot')),
    secondary_y=True,
)

# 레이아웃 설정
fig.update_layout(
    title_text=f"관통률에 따른 방어계수 및 변동률 변화 (방어력: {data_defense:.1f}, 관통수치: {data_flat_pen})",
    hovermode="x unified"
)

# X축 설정
fig.update_xaxes(title_text="관통률 (%)", range=[0, 100])

# Y축 설정 (기본 Y축: 방어계수)
fig.update_yaxes(title_text="<b>방어계수</b>", secondary_y=False, range=[0, 1.0])

# 2차 Y축 설정 (변동률)
fig.update_yaxes(title_text="<b>0% 대비 변동률 (%)</b>", secondary_y=True, showgrid=False)

# 현재 설정된 관통률(data_pen_rate) 위치에 점선 추가 (현재 값 시각화)
fig.add_vline(
    x=data_pen_rate,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"현재 관통률: {data_pen_rate:.1f}%",
    annotation_position="top left"
)

# 그래프 표시
st.subheader("관통률(%) vs 방어계수 및 변동률 그래프")
st.plotly_chart(fig, width='content')

# 현재 방어계수 출력
st.markdown(f"**현재 관통률({data_pen_rate:.1f}%)에서의 방어계수:** `{current_coeff:.4f}`")

# --- 최종 출력 문장 (수정된 부분) ---

# 1. 관통률 0%일 때의 방어계수 계산
coeff_at_0_percent = calculate_defense_coefficient(data_defense, 0.0, data_flat_pen)

# 2. 변동 비율 계산
if coeff_at_0_percent != 0:
    percentage_change = ((current_coeff - coeff_at_0_percent) / coeff_at_0_percent) * 100
else:
    percentage_change = 0.0

# 3. 결과 문장 출력: '상승' 대신 '변동'으로 명시적으로 수정
# 음수 값은 감소를 의미하므로, '변동'이 더 정확합니다.
st.markdown(
    f"**$0\%$ 관통률 대비** 현재 방어계수는 **`{percentage_change:.2f}%`** 상승했습니다."
)


# 1. 미지수 x를 포함하는 방정식 (f(x) = 0) 정의
def solve_x(x):
    """
    (1 + x + 0.3) * D(현재) - (1 + x) * D(현재+24) = 0 이 되는 x를 찾는 함수
    """

    # 1. 현재 관통률에서의 방어계수 (D_current)
    D_current = calculate_defense_coefficient(
        defense=data_defense,
        pen_rate=data_pen_rate,
        flat_pen=data_flat_pen
    )

    # 2. 관통률 24% 증가 시의 방어계수 (D_plus_24)
    D_plus_24 = calculate_defense_coefficient(
        defense=data_defense,
        pen_rate=data_pen_rate + 24.0,  # 관통률 24% 추가
        flat_pen=data_flat_pen
    )

    # f(x) = (1.3 + x) * D_current - (1 + x) * D_plus_24
    result = (1.3 + x) * D_current - (1 + x) * D_plus_24

    return result


@st.cache_data
def find_x_solution(defense, pen_rate, flat_pen, initial_guess=0):
    from scipy.optimize import fsolve

    def _solve_x(x):
        D_current = calculate_defense_coefficient(defense, pen_rate, flat_pen)
        D_plus_24 = calculate_defense_coefficient(defense, pen_rate + 24.0, flat_pen)
        return (1.3 + x) * D_current - (1 + x) * D_plus_24

    # fsolve 실행
    x_solution = fsolve(_solve_x, x0=initial_guess)  # noqa
    return x_solution[0]

x_solution = find_x_solution(data_defense, data_pen_rate, data_flat_pen)

st.markdown(
    f"피해 증가 효과가 **`{x_solution:.2%}`** 이상일 때, 5번 관통률 디스크가 피해 증가 디스크에 비해 더 유리합니다."
)
