# SVR 기반 재생에너지 예측 오차 정량화를 활용한 VPP-ESS 출력제어 저감 시뮬레이션

이 저장소는 제주도 태양광·풍력 발전량, 제주 Open-Meteo 시간별 기상자료, KPX 5분 수요예측 자료를 결합하여 단순화된 14노드 / 20선로 VPP-ESS 제어 시뮬레이션을 수행한다. 기존의 합성 재생에너지·부하 시나리오 기반 코드를 유지하되, 기본 실행은 실제 CSV 자료를 사용하며, SVR 예측 모델의 보정기간 잔차 분위수로 불확실성 폭을 산정해 ESS 출력제어 저감 효과를 비교한다.

## 연구 질문

본 연구의 핵심 질문은 다음과 같다.

“SVR 예측 잔차의 분위수로 산정한 불확실성 폭을 ESS 제어에 반영하면, 경험적 구름량 기반 불확실성 방식보다 재생에너지 출력제어를 줄일 수 있는가?”

즉 단순히 재생에너지 발전량의 평균 예측값만 사용하는 것이 아니라, 예측 오차의 크기를 별도 보정기간에서 추정한 뒤 ESS 충방전 판단에 반영하는 방식이 출력제어량을 줄이는지 확인한다.

## 데이터셋 설명

### KPX 지역별 시간별 태양광 및 풍력 발전량

역할: 제주도 재생에너지 실제 발전량 입력과 SVR 예측 목표값으로 사용한다.

주요 열: `거래일`, `거래시간`, `지역`, `연료원`, `전력거래량(MWh)`

전처리: `지역 == 제주도`인 행만 사용하고, `연료원`은 `태양광`과 `풍력`을 모두 사용한다. `거래시간`은 1~24 형식이므로 `거래시간 1 = 00:00~01:00`으로 보고 `hour = 거래시간 - 1`로 변환한다. 같은 시각의 태양광과 풍력 발전량을 합산해 `renewable_mwh_raw`를 만든다.

한계: KPX 거래자료는 연구에 사용할 수 있는 공개 자료이지만, 제주도 내 모든 분산형 설비의 실제 물리 출력 전체를 완전히 대표한다고 단정할 수 없다.

### KPX 5분 수요예측 자료

역할: 시뮬레이션의 시간별 부하 입력으로 사용한다.

주요 열: `시간`, `수요예측MW`

전처리: CSV 앞의 설명 행을 건너뛰고 실제 헤더부터 읽는다. `시간`을 datetime으로 변환하며, `2025-03-01`처럼 날짜만 있는 행은 자정 시각으로 해석한다. 5분 단위 예측값은 시간별 평균으로 집계해 `load_mw_raw`를 만든다.

한계: 수요예측 자료는 전국 또는 송전단 기준 수요 규모일 수 있으므로, 원자료 MW를 장난감 14버스 계통에 직접 주입하지 않고 고정 스케일링을 적용한다.

### Open-Meteo 제주 시간별 기상자료

역할: SVR 예측 입력 특징으로 사용한다.

주요 열: `time`, `temperature_2m`, `relative_humidity_2m`, `precipitation`, `cloud_cover`, `shortwave_radiation`, `wind_speed_10m`, `wind_direction_10m`

전처리: 메타데이터 행을 건너뛰고 첫 번째 열이 `time`인 행을 헤더로 탐지한다. 괄호 안 단위 표기를 제거해 열 이름을 정규화한다. 풍향은 sin/cos 순환 특징으로 변환한다.

한계: Open-Meteo 자료는 특정 발전소 현장의 계측 센서값이 아니므로 실제 발전단지의 미세 기상과 차이가 있을 수 있다.

### 육지 출력제어횟수 CSV

일부 원자료 폴더에는 육지 태양광·풍력 출력제어횟수 CSV가 포함될 수 있다. 본 연구는 제주 기상 및 제주 재생에너지 발전량을 사용하므로, 육지 출력제어횟수 데이터는 직접 검증 지표로 사용하지 않았다. 지역이 맞지 않는 자료로 제주 시뮬레이션을 검증했다고 주장하지 않는다.

## 폴더 구조

```text
data/raw/                 실제 CSV 원자료 위치
outputs/data/             모델링 테이블, 통계, 로그 CSV 저장 위치
outputs/figures/          연구 보고서용 그림 저장 위치
simulate_vpp.py           전체 시뮬레이션 실행 스크립트
requirements.txt          필요한 파이썬 패키지 목록
```

## 실행 방법

패키지 설치:

```bash
python -m pip install -r requirements.txt
```

실제 데이터 기반 기본 실행:

```bash
python simulate_vpp.py
```

기존 합성 시나리오 fallback 실행:

```bash
python simulate_vpp.py --synthetic
```

실제 데이터 모드에서 필요한 CSV가 없으면 자동으로 합성 모드로 넘어가지 않는다. 누락된 자료 종류를 출력하고 `data/raw/`에 CSV를 넣으라는 오류를 낸다.

## 고정 연구 설정

본 프로젝트는 일반 목적 CLI가 아니라 단일 연구 설정을 재현하기 위한 코드이다. 다음 값은 코드 상단 상수로 고정되어 있다.

| 항목 | 값 |
|---|---|
| 지역 | 제주도 |
| 연료 | 태양광 + 풍력 합산 |
| 전체 분석 기간 | 2025-03-01 ~ 2025-05-31 |
| SVR 학습 기간 | 2025-03-01 ~ 2025-04-15 |
| 불확실성 보정 기간 | 2025-04-16 ~ 2025-04-30 |
| 최종 테스트 / 시뮬레이션 기간 | 2025-05-01 ~ 2025-05-31 |
| 재생에너지 toy-grid 기준 용량 | `CAP_RENEWABLE = 200.0` |
| 부하 toy-grid 기준 평균 | `LOAD_TARGET_MEAN = 175.0` |

## 데이터 전처리

KPX 발전량 자료의 `거래시간`은 1~24를 0~23시로 변환한다. `거래시간 1`은 00:00~01:00 구간으로 해석한다.

KPX 5분 수요예측 자료는 시간별 평균으로 변환한다. 이 값은 원자료 추적을 위해 `load_mw_raw`로 저장하고, 실제 DC power flow에는 학습기간 평균 기반 스케일링 후 `load_scaled_mw`를 사용한다.

Open-Meteo 자료는 메타데이터 행을 자동으로 건너뛰고 `time` 헤더 행부터 읽는다. 열 이름의 단위 괄호를 제거해 SVR 입력에 필요한 표준 열 이름으로 정규화한다.

세 자료는 `datetime` 기준으로 병합한다. 필수 목표값, 부하, 기상 입력이 빠진 행은 제외하며, 제외 사유는 `outputs/data/exclusion_log.csv`에 기록한다. 출력 로그에는 재생에너지 행 수, 수요 행 수, 기상 행 수, 병합 행 수, 결측 제외 행 수, 최종 유효 행 수가 표시된다.

## Toy Grid 스케일링

KPX 원자료의 발전량과 수요는 단순화된 14노드 toy grid에 직접 넣기에는 규모가 맞지 않는다. 따라서 학습기간만 사용해 고정 스케일링 계수를 계산한다.

재생에너지 스케일링:

```text
renewable_p95_raw = 학습기간 renewable_mwh_raw의 95분위
renewable_scale_factor = CAP_RENEWABLE / renewable_p95_raw
renewable_scaled_mw = renewable_mwh_raw * renewable_scale_factor
```

부하 스케일링:

```text
load_mean_raw = 학습기간 load_mw_raw 평균
load_scale_factor = LOAD_TARGET_MEAN / load_mean_raw
load_scaled_mw = load_mw_raw * load_scale_factor
```

같은 재생에너지 스케일링 계수를 실제 발전량, SVR 예측값, SVR 잔차 분위수 불확실성에 모두 적용한다. 스케일링 상수와 주요 가정은 `outputs/data/assumptions.json`에 저장된다.

## SVR 예측 방법

모델은 scikit-learn Pipeline으로 구성된다.

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))
])
```

입력 특징은 시간 순환 특징, 계절 순환 특징, 기상 특징, 풍향 순환 특징, 재생에너지 lag/rolling 특징이다.

사용 특징:

```text
hour_sin, hour_cos
doy_sin, doy_cos
temperature_2m, relative_humidity_2m, precipitation, cloud_cover
shortwave_radiation, wind_speed_10m
wind_direction_sin, wind_direction_cos
renewable_lag_1h, renewable_lag_24h
renewable_rolling_mean_24h, renewable_rolling_std_24h
```

목표값은 `renewable_mwh_raw`이다.

자료 분할은 시간 순서를 유지한다. 무작위 분할은 사용하지 않는다.

| 구간 | 용도 |
|---|---|
| 2025-03-01 ~ 2025-04-15 | SVR 학습 |
| 2025-04-16 ~ 2025-04-30 | 잔차 분위수 보정 |
| 2025-05-01 ~ 2025-05-31 | 최종 알고리즘 비교 |

## 불확실성 산정

SVR의 support-vector 비율은 확률 보장으로 사용하지 않는다. SVR의 `epsilon` 값 자체도 확률적 신뢰구간을 보장하지 않는다.

본 연구의 불확실성 폭은 별도 보정기간의 실제 예측 오차에서 계산한다.

```text
residual_calib = actual_calib_raw - forecast_calib_raw
abs_error_calib = abs(residual_calib)
epsilon_80_raw = abs_error_calib의 80분위
epsilon_90_raw = abs_error_calib의 90분위
epsilon_95_raw = abs_error_calib의 95분위
```

최종 시뮬레이션에는 시간대별 90분위 절대오차를 사용한다.

```text
epsilon_90_by_hour = 보정기간 abs_error를 hour별 그룹화한 90분위
sigma_u_raw_mwh = 해당 hour의 epsilon_90_by_hour
sigma_u_scaled_mw = sigma_u_raw_mwh * renewable_scale_factor
```

특정 시간대의 보정 샘플이 너무 적거나 분위수를 계산할 수 없으면 전체 보정기간의 `epsilon_90_raw`로 대체한다. 대체 횟수는 `svr_metrics.csv`와 `data_warnings.txt`에 기록한다.

## Dispatch 모드

비교 대상은 다섯 가지이다.

`Heuristic_Rule`: 낮 시간대에는 고정 규칙으로 충전하고 저녁 시간대에는 방전한다. 고급 예측 불확실성은 사용하지 않는다.

`Deterministic_Opt`: SVR 평균 예측값을 신뢰하고 하루 ESS 계획을 만든다. 추가 불확실성 margin은 넣지 않는다.

`Rolling_Greedy`: 현재 실제 재생에너지, 현재 부하, 현재 선로 혼잡에 즉시 반응한다. 미래 불확실성 margin은 사용하지 않는다.

`Stochastic_Proposed`: SVR 예측값과 구름량 기반 경험적 불확실성을 함께 사용한다. 구름량은 `cloud_cover / 100`으로 정규화하고 다음 식을 사용한다.

```text
sigma_heuristic_scaled_mw =
forecast_renewable_scaled_mw * (0.10 + 0.25 * cloud_norm)
```

`SVR_ResidualQuantile`: SVR 예측값과 보정기간 잔차의 시간대별 90분위 불확실성을 사용한다. 이 방식이 본 연구의 데이터 기반 불확실성 제어 방식이다.

두 불확실성 기반 모드는 같은 형태의 stochastic objective dispatch 함수를 사용하지만, 넘겨주는 `sigma`가 다르다. `Stochastic_Proposed`는 구름량 경험식, `SVR_ResidualQuantile`은 보정기간 잔차 분위수를 사용한다.

## Main 분석과 Sensitivity 분석

Main 분석은 2025년 5월의 모든 유효 날짜를 사용한다. 유효 날짜는 해당 날짜에 필요한 실제 재생에너지, SVR 예측, 불확실성, 부하, 기상 파생 특징이 24시간 모두 존재하는 날짜이다.

Top-risk 날짜 10개는 별도 sensitivity 분석으로만 사용한다. 이 날짜들은 다음 일별 risk score로 선택한다.

```text
daily_risk = max(renewable_scaled_mw - 0.86 * load_scaled_mw)
```

이 값이 모두 음수 등으로 유용하지 않으면 `renewable_scaled_mw / load_scaled_mw`의 일 최대값을 fallback으로 사용한다.

결론은 all valid May dates를 사용하는 main 분석을 중심으로 해석한다. Top-risk 분석은 고위험 날짜에서 경향이 어떻게 달라지는지 보는 보조 분석이며, 주 결론의 근거로 사용하지 않는다. 이는 유리한 날짜만 고르는 cherry-picking을 방지하기 위한 설계이다.

## 연구 윤리 및 조작 방지 정책

본 저장소는 다음 규칙을 따른다.

고정 상수는 알고리즘 비교 전에 정한다. `CAP_RENEWABLE`, `LOAD_TARGET_MEAN`, ESS 용량, ESS 출력 한계, 선로 한계, risk-z, dispatch 상수는 결과를 본 뒤 조정하지 않는다.

Main 결과는 유효한 2025년 5월 날짜 전체를 사용한다. 특정 알고리즘이 잘 나오는 날짜만 골라 결론을 내리지 않는다.

불확실성 보정은 2025-04-16 ~ 2025-04-30 기간만 사용한다. 2025년 5월 최종 테스트 자료는 `epsilon` 또는 `sigma_u` 계산에 사용하지 않는다.

모든 제외 행 또는 제외 날짜는 `outputs/data/exclusion_log.csv`에 기록한다. 제외는 결측, 중복, lag 특징 부족 같은 데이터 품질 사유로만 수행한다.

고정 가정과 스케일링 계수는 `outputs/data/assumptions.json`에 저장한다.

`outputs/data/data_warnings.txt`에는 데이터 한계, 제외 날짜 수, 최종 테스트 날짜가 모든 유효 5월 날짜인지 여부, top-risk 분석이 sensitivity-only라는 설명, 육지 출력제어횟수 자료를 직접 검증에 쓰지 않았다는 문구를 저장한다.

육지 출력제어횟수 CSV는 제주 시뮬레이션의 직접 검증 지표로 사용하지 않는다. 본 연구는 제주 기상과 제주 재생에너지 발전량을 사용하기 때문이다.

## 출력 파일

### CSV 및 JSON

`outputs/data/modeling_table.csv`: 재생에너지, 부하, 기상, SVR 특징, 예측값, 스케일링 결과가 병합된 시간별 모델링 테이블

`outputs/data/svr_metrics.csv`: SVR 학습/보정/테스트 샘플 수, MAE, RMSE, R2, 보정 잔차 분위수, 스케일링 계수

`outputs/data/main_simulation_dates.csv`: main 분석에 사용한 모든 유효 2025년 5월 날짜

`outputs/data/selected_risk_dates_sensitivity.csv`: top-risk sensitivity 분석 날짜

`outputs/data/exclusion_log.csv`: 제외된 행 또는 날짜와 사유

`outputs/data/assumptions.json`: 고정 연구 설정, 상수, feature set, scaling factor

`outputs/data/data_warnings.txt`: 데이터 한계와 분석 경고

`outputs/data/time_series.csv`: subject/date/hour/mode별 시간대 시뮬레이션 결과

`outputs/data/line_flows.csv`: 20개 선로의 시간대별 조류와 이용률

`outputs/data/simulation_summary.csv`: subject/date/mode별 요약 성능

`outputs/data/repeated_measures_anova.csv`: 날짜를 반복측정 subject로 둔 ANOVA 결과

`outputs/data/paired_tests.csv`: baseline 모드와 제안 모드의 대응 비교 검정

### Figures

`outputs/figures/figure1_curtailment_bar.png`: main 분석 전체 유효 5월 날짜의 알고리즘별 출력제어량 평균 및 표준편차

`outputs/figures/figure2_soc_timeseries.png`: median-risk 대표 날짜의 ESS SOC 변화

`outputs/figures/figure3_uncertainty_risk.png`: 대표 날짜의 실제 재생에너지, SVR 예측, 구름량 기반 밴드, SVR 잔차 분위수 밴드

`outputs/figures/figure4_line_heatmap.png`: 대표 날짜 `SVR_ResidualQuantile` 모드의 선로 이용률 heatmap

`outputs/figures/figure5_svr_forecast_interval.png`: 대표 날짜의 SVR 예측과 hourly epsilon90 구간

`outputs/figures/figure6_daily_surplus_risk.png`: 모든 유효 5월 날짜의 일별 surplus risk와 top-risk sensitivity 날짜 표시

Figure 7이나 육지 출력제어횟수 기반 직접 검증 그림은 만들지 않는다.

## 통계 분석

반복측정 구조에서 subject는 날짜이다. 같은 날짜에 모든 알고리즘을 적용하므로 날짜별 기상·수요 난이도 차이를 통제한 비교가 가능하다.

Main 통계는 `analysis_type == main_all_valid_may`만 사용한다. Sensitivity 통계는 `analysis_type == sensitivity_top_risk`로 별도 저장하며 main 분석과 섞지 않는다.

반복측정 ANOVA는 `total_curtailment_mwh`에 대해 모드 효과를 계산한다. 대응 비교 검정은 다음 baseline과 proposed 조합을 계산한다.

```text
Heuristic_Rule vs Stochastic_Proposed
Deterministic_Opt vs Stochastic_Proposed
Rolling_Greedy vs Stochastic_Proposed
Heuristic_Rule vs SVR_ResidualQuantile
Deterministic_Opt vs SVR_ResidualQuantile
Rolling_Greedy vs SVR_ResidualQuantile
```

유효 main 날짜가 5개 미만이면 시뮬레이션은 계속 실행하지만 `data_warnings.txt`에 통계 신뢰도 경고를 남긴다.

## 방법론적 한계

14노드 / 20선로 DC power flow 계통은 실제 한국 전력망이 아니다. 실제 계통 운영 결과를 예측하는 모델이 아니라, 동일한 toy-grid 제약에서 ESS 제어 알고리즘의 상대적 차이를 비교하는 연구용 모델이다.

원자료 발전량과 수요는 toy grid에 맞게 스케일링되므로 절대 출력제어량보다 알고리즘 간 상대 비교를 중심으로 해석해야 한다.

Open-Meteo 기상자료는 실제 발전소 현장 계측값과 다를 수 있다.

KPX 공개 자료는 연구에 사용할 수 있는 거래·예측 자료이지만 모든 분산형 발전 설비와 모든 운영상황을 완전히 대표한다고 볼 수 없다.

육지 출력제어횟수 자료는 제주 기상·제주 발전량 기반 시뮬레이션의 직접 검증 지표로 사용하지 않는다.

SVR support-vector 비율이나 SVR epsilon은 확률 보장이 아니다. 본 연구의 불확실성 폭은 별도 보정기간 잔차 분위수에서만 산정된다.

## 재현성 체크리스트

- 분석 기간, 학습 기간, 보정 기간, 테스트 기간이 고정되어 있다.
- 지역은 제주도, 연료는 태양광 + 풍력 합산으로 고정되어 있다.
- `CAP_RENEWABLE`, `LOAD_TARGET_MEAN`, ESS 용량, 선로 한계, risk-z가 코드 상단에 고정되어 있다.
- SVR feature set이 고정되어 있다.
- 무작위 train/test split을 사용하지 않는다.
- 2025년 5월 테스트 자료를 불확실성 보정에 사용하지 않는다.
- Main 분석은 모든 유효 5월 날짜를 사용한다.
- 제외 기록은 `exclusion_log.csv`에 저장된다.
- 고정 가정은 `assumptions.json`에 저장된다.
- 데이터 한계와 경고는 `data_warnings.txt`에 저장된다.
