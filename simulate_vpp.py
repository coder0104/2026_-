from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Fixed real-data research setting. Do not tune these after seeing results.
DATA_DIR = Path("data/raw")
REGION_FIXED = "제주도"
FUEL_MODE_FIXED = "solar_wind_combined"
ANALYSIS_START = "2025-03-01"
ANALYSIS_END = "2025-05-31"
TRAIN_START = "2025-03-01"
TRAIN_END = "2025-04-15"
CALIB_START = "2025-04-16"
CALIB_END = "2025-04-30"
TEST_START = "2025-05-01"
TEST_END = "2025-05-31"

TIME_HORIZON = 24
DT = 1.0
N_NODES = 14
LINE_TRIP_LIMIT = 80.0
CAP_RENEWABLE = 200.0
LOAD_TARGET_MEAN = 175.0
ESS_TOTAL_CAP = 400.0
ESS_EFF = 0.95
N_SEEDS = 10

SYNTHETIC_MODES = [
    "Heuristic_Rule",
    "Deterministic_Opt",
    "Rolling_Greedy",
    "Stochastic_Proposed",
]
REAL_MODES = SYNTHETIC_MODES + ["SVR_ResidualQuantile"]

OUT_DIR = Path("outputs")
OUTPUT_DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

LINE_X = np.array(
    [
        0.06,
        0.08,
        0.05,
        0.07,
        0.06,
        0.08,
        0.04,
        0.09,
        0.10,
        0.07,
        0.08,
        0.06,
        0.09,
        0.05,
        0.07,
        0.06,
        0.08,
        0.07,
        0.05,
        0.06,
    ]
)
LINES = np.array(
    [
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (3, 4),
        (3, 6),
        (3, 8),
        (4, 5),
        (5, 10),
        (5, 11),
        (5, 12),
        (6, 7),
        (6, 8),
        (8, 9),
        (8, 13),
        (9, 10),
        (11, 12),
        (12, 13),
    ],
    dtype=int,
)

RENEWABLE_NODES = np.array([2, 5, 8, 11])
ESS_NODES = np.array([2, 5, 8, 11])
ESS_NODE_CAP = ESS_TOTAL_CAP / len(ESS_NODES)
ESS_NODE_POWER = 55.0
LOCAL_RENEWABLE_EXPORT_LIMIT = 42.0
STOCHASTIC_LOOKAHEAD = 5
STOCHASTIC_RISK_Z = 1.15

LOAD_WEIGHTS = np.array(
    [0.02, 0.05, 0.07, 0.10, 0.06, 0.08, 0.09, 0.05, 0.11, 0.08, 0.09, 0.07, 0.08, 0.05]
)
LOAD_WEIGHTS = LOAD_WEIGHTS / LOAD_WEIGHTS.sum()
RENEWABLE_WEIGHTS = np.array([0.12, 0.16, 0.20, 0.52])

SVR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "renewable_lag_1h",
    "renewable_lag_24h",
    "renewable_rolling_mean_24h",
    "renewable_rolling_std_24h",
]
REQUIRED_RAW_COLUMNS = [
    "solar_mwh_raw",
    "wind_mwh_raw",
    "renewable_mwh_raw",
    "load_mw_raw",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
]


@dataclass(frozen=True)
class Scenario:
    forecast_renewable: np.ndarray
    actual_renewable: np.ndarray
    sigma_u: np.ndarray
    load_by_node: np.ndarray
    renewable_share_by_node: np.ndarray
    sigma_heuristic: np.ndarray | None = None
    actual_renewable_raw: np.ndarray | None = None
    forecast_renewable_raw: np.ndarray | None = None
    sigma_u_raw: np.ndarray | None = None
    total_load_raw: np.ndarray | None = None
    total_load_scaled: np.ndarray | None = None
    date: str | None = None
    subject_id: str | None = None
    analysis_type: str = "synthetic"
    hourly_frame: pd.DataFrame | None = field(default=None, compare=False)


def configure_korean_font() -> None:
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def build_dc_matrix() -> tuple[np.ndarray, np.ndarray]:
    incidence = np.zeros((len(LINES), N_NODES))
    for idx, (a, b) in enumerate(LINES):
        incidence[idx, a] = 1.0
        incidence[idx, b] = -1.0
    susceptance = np.diag(1.0 / LINE_X)
    b_matrix = incidence.T @ susceptance @ incidence
    return incidence, b_matrix


INCIDENCE, B_MATRIX = build_dc_matrix()


def dc_power_flow(injection: np.ndarray) -> np.ndarray:
    balanced = injection.copy()
    balanced[0] -= balanced.sum()
    reduced_b = B_MATRIX[1:, 1:]
    theta = np.zeros(N_NODES)
    theta[1:] = np.linalg.solve(reduced_b, balanced[1:])
    return (INCIDENCE @ theta) / LINE_X


def ensure_output_dirs() -> None:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", str(value))


def read_csv_with_encodings(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV with supported encodings: {path}") from last_error


def find_files(keyword_parts: list[str]) -> list[Path]:
    files = []
    for path in DATA_DIR.glob("**/*.csv"):
        nfc_name = normalize_text(path.name)
        if all(part in nfc_name for part in keyword_parts):
            files.append(path)
    return sorted(files)


def require_real_data_files() -> None:
    missing = []
    if not find_files(["5분", "수요", "2025년03월"]):
        missing.append("KPX 5-minute demand forecast CSVs for 2025-03 to 2025-05")
    if not find_files(["지역별", "태양광", "풍력", "발전량"]):
        missing.append("KPX regional hourly solar/wind generation CSV")
    if not find_files(["open-meteo"]):
        missing.append("Open-Meteo Jeju hourly weather CSV")
    if missing:
        print("Required real-data files are missing.")
        for item in missing:
            print(f"- {item}")
        print("Place the CSV files in data/raw/ or run synthetic fallback with: python simulate_vpp.py --synthetic")
        raise FileNotFoundError("Missing required real-data CSV files.")


def read_demand_files() -> pd.DataFrame:
    files = [p for p in find_files(["5분", "수요"]) if any(f"2025년{m:02d}월" in normalize_text(p.name) for m in [3, 4, 5])]
    if not files:
        raise FileNotFoundError("No KPX 5-minute demand forecast files found.")
    frames = []
    for path in files:
        df = read_csv_with_encodings(path, skiprows=2)
        df.columns = [normalize_text(c).strip() for c in df.columns]
        if "시간" not in df.columns or "수요예측MW" not in df.columns:
            raise ValueError(f"Unexpected demand columns in {path}: {df.columns.tolist()}")
        df = df[["시간", "수요예측MW"]].copy()
        df["datetime"] = pd.to_datetime(df["시간"], errors="coerce", format="mixed")
        df["load_mw_raw"] = pd.to_numeric(df["수요예측MW"], errors="coerce")
        frames.append(df[["datetime", "load_mw_raw"]])
    demand = pd.concat(frames, ignore_index=True).dropna(subset=["datetime"])
    demand = demand.sort_values("datetime")
    demand = demand.groupby("datetime", as_index=False)["load_mw_raw"].mean()
    hourly = demand.set_index("datetime").resample("h")["load_mw_raw"].mean().reset_index()
    hourly = hourly[(hourly["datetime"] >= ANALYSIS_START) & (hourly["datetime"] <= f"{ANALYSIS_END} 23:00:00")]
    print(f"Demand rows loaded: {len(hourly)} hourly rows from {len(files)} files")
    return hourly


def read_renewable_file(exclusion_rows: list[dict]) -> pd.DataFrame:
    files = find_files(["지역별", "태양광", "풍력", "발전량"])
    if not files:
        raise FileNotFoundError("No KPX regional solar/wind generation CSV found.")
    df = read_csv_with_encodings(files[0])
    df.columns = [normalize_text(c).strip() for c in df.columns]
    required = ["거래일", "거래시간", "지역", "연료원", "전력거래량(MWh)"]
    if any(col not in df.columns for col in required):
        raise ValueError(f"Unexpected renewable columns in {files[0]}: {df.columns.tolist()}")
    df = df[required].copy()
    df["지역"] = df["지역"].map(normalize_text)
    df["연료원"] = df["연료원"].map(normalize_text)
    df = df[(df["지역"] == REGION_FIXED) & (df["연료원"].isin(["태양광", "풍력"]))]
    df["date_only"] = pd.to_datetime(df["거래일"], errors="coerce")
    df["hour"] = pd.to_numeric(df["거래시간"], errors="coerce") - 1
    df["value"] = pd.to_numeric(df["전력거래량(MWh)"], errors="coerce")
    df = df.dropna(subset=["date_only", "hour", "value"])
    df["datetime"] = df["date_only"] + pd.to_timedelta(df["hour"].astype(int), unit="h")
    duplicate_mask = df.duplicated(["datetime", "연료원"], keep=False)
    for dt in df.loc[duplicate_mask, "datetime"].drop_duplicates():
        exclusion_rows.append(
            {
                "datetime": dt,
                "date": pd.Timestamp(dt).date().isoformat(),
                "reason": "duplicate_timestamp_removed",
                "source_stage": "renewable",
            }
        )
    pivot = (
        df.groupby(["datetime", "연료원"], as_index=False)["value"]
        .sum()
        .pivot(index="datetime", columns="연료원", values="value")
        .rename(columns={"태양광": "solar_mwh_raw", "풍력": "wind_mwh_raw"})
        .reset_index()
    )
    for col in ["solar_mwh_raw", "wind_mwh_raw"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot[["solar_mwh_raw", "wind_mwh_raw"]] = pivot[["solar_mwh_raw", "wind_mwh_raw"]].fillna(0.0)
    pivot["renewable_mwh_raw"] = pivot["solar_mwh_raw"] + pivot["wind_mwh_raw"]
    pivot = pivot[(pivot["datetime"] >= ANALYSIS_START) & (pivot["datetime"] <= f"{ANALYSIS_END} 23:00:00")]
    print(f"Renewable rows loaded: {len(pivot)} hourly rows")
    return pivot[["datetime", "solar_mwh_raw", "wind_mwh_raw", "renewable_mwh_raw"]]


def read_weather_file() -> pd.DataFrame:
    files = find_files(["open-meteo"])
    if not files:
        raise FileNotFoundError("No Open-Meteo weather CSV found.")
    path = files[0]
    header_row = 0
    for encoding in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            with path.open(encoding=encoding) as handle:
                for idx, line in enumerate(handle):
                    first = line.split(",", 1)[0].strip()
                    if first == "time":
                        header_row = idx
                        raise StopIteration
        except StopIteration:
            break
        except UnicodeDecodeError:
            continue
    df = read_csv_with_encodings(path, skiprows=header_row)
    normalized = []
    for col in df.columns:
        base = normalize_text(col).strip()
        base = re.sub(r"\s*\(.*?\)", "", base).strip()
        normalized.append(base)
    df.columns = normalized
    expected = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "shortwave_radiation",
        "wind_speed_10m",
        "wind_direction_10m",
    ]
    if any(col not in df.columns for col in expected):
        raise ValueError(f"Unexpected weather columns in {path}: {df.columns.tolist()}")
    df = df[expected].copy()
    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    for col in expected[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df[(df["datetime"] >= ANALYSIS_START) & (df["datetime"] <= f"{ANALYSIS_END} 23:00:00")]
    print(f"Weather rows loaded: {len(df)} hourly rows")
    return df.drop(columns=["time"])


def add_exclusion(exclusion_rows: list[dict], rows: pd.DataFrame, reason: str, source_stage: str) -> None:
    for dt in rows["datetime"].dropna().drop_duplicates():
        ts = pd.Timestamp(dt)
        exclusion_rows.append(
            {
                "datetime": ts,
                "date": ts.date().isoformat(),
                "reason": reason,
                "source_stage": source_stage,
            }
        )


def build_modeling_table(
    renewable: pd.DataFrame,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    exclusion_rows: list[dict],
) -> pd.DataFrame:
    timeline = pd.DataFrame({"datetime": pd.date_range(ANALYSIS_START, f"{ANALYSIS_END} 23:00:00", freq="h")})
    merged = timeline.merge(renewable, on="datetime", how="left").merge(demand, on="datetime", how="left").merge(
        weather, on="datetime", how="left"
    )
    print(f"Merged rows before exclusions: {len(merged)}")

    missing_any = pd.Series(False, index=merged.index)
    reason_by_column = {
        "renewable_mwh_raw": "missing_renewable",
        "load_mw_raw": "missing_load",
        "temperature_2m": "missing_weather",
        "relative_humidity_2m": "missing_weather",
        "precipitation": "missing_weather",
        "cloud_cover": "missing_weather",
        "shortwave_radiation": "missing_weather",
        "wind_speed_10m": "missing_weather",
        "wind_direction_10m": "missing_weather",
    }
    for col, reason in reason_by_column.items():
        missing = merged[col].isna()
        if missing.any():
            add_exclusion(exclusion_rows, merged.loc[missing, ["datetime"]], reason, "merge")
        missing_any |= missing

    dropped_missing = int(missing_any.sum())
    valid = merged.loc[~missing_any].copy()
    valid["date"] = valid["datetime"].dt.date.astype(str)
    valid["hour"] = valid["datetime"].dt.hour
    valid = add_svr_features(valid, exclusion_rows)
    print(f"Rows dropped due to missing values: {dropped_missing}")
    print(f"Final valid rows after feature engineering: {len(valid)}")
    return valid


def add_svr_features(df: pd.DataFrame, exclusion_rows: list[dict]) -> pd.DataFrame:
    df = df.sort_values("datetime").copy()
    hour = df["datetime"].dt.hour
    doy = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["wind_direction_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
    df["wind_direction_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)
    df["renewable_lag_1h"] = df["renewable_mwh_raw"].shift(1)
    df["renewable_lag_24h"] = df["renewable_mwh_raw"].shift(24)
    rolling = df["renewable_mwh_raw"].shift(1).rolling(24, min_periods=24)
    df["renewable_rolling_mean_24h"] = rolling.mean()
    df["renewable_rolling_std_24h"] = rolling.std()
    feature_missing = df[SVR_FEATURES].isna().any(axis=1)
    if feature_missing.any():
        add_exclusion(exclusion_rows, df.loc[feature_missing, ["datetime"]], "insufficient_lag_feature", "feature_engineering")
    return df.loc[~feature_missing].copy()


def train_svr_forecaster(modeling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = modeling.copy()
    train_mask = (df["datetime"] >= TRAIN_START) & (df["datetime"] <= f"{TRAIN_END} 23:00:00")
    calib_mask = (df["datetime"] >= CALIB_START) & (df["datetime"] <= f"{CALIB_END} 23:00:00")
    test_mask = (df["datetime"] >= TEST_START) & (df["datetime"] <= f"{TEST_END} 23:00:00")

    train = df.loc[train_mask].copy()
    calib = df.loc[calib_mask].copy()
    test = df.loc[test_mask].copy()
    if train.empty or calib.empty or test.empty:
        raise ValueError("Train, calibration, and test splits must all contain valid rows.")

    renewable_p95_raw = float(train["renewable_mwh_raw"].quantile(0.95))
    load_mean_raw = float(train["load_mw_raw"].mean())
    if renewable_p95_raw <= 0 or load_mean_raw <= 0:
        raise ValueError("Training-period scaling denominators must be positive.")
    renewable_scale_factor = CAP_RENEWABLE / renewable_p95_raw
    load_scale_factor = LOAD_TARGET_MEAN / load_mean_raw

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        ]
    )
    model.fit(train[SVR_FEATURES], train["renewable_mwh_raw"])

    for mask_name, mask in [("train", train_mask), ("calibration", calib_mask), ("test", test_mask)]:
        pred = np.clip(model.predict(df.loc[mask, SVR_FEATURES]), 0, None)
        df.loc[mask, "forecast_renewable_raw_mwh"] = pred
        print(f"SVR predicted {mask_name} rows: {len(pred)}")

    calib_pred = df.loc[calib_mask, "forecast_renewable_raw_mwh"]
    calib_actual = df.loc[calib_mask, "renewable_mwh_raw"]
    abs_error_calib = (calib_actual - calib_pred).abs()
    epsilon_80_raw = float(abs_error_calib.quantile(0.80))
    epsilon_90_raw = float(abs_error_calib.quantile(0.90))
    epsilon_95_raw = float(abs_error_calib.quantile(0.95))

    hourly_quantile = abs_error_calib.groupby(df.loc[calib_mask, "hour"]).quantile(0.90)
    hourly_count = abs_error_calib.groupby(df.loc[calib_mask, "hour"]).count()
    fallback_hours = set(hourly_count[hourly_count < 5].index.tolist())
    fallback_count = 0
    sigma_values = []
    for _, row in df.iterrows():
        hour = int(row["hour"])
        if hour in hourly_quantile.index and hour not in fallback_hours and pd.notna(hourly_quantile.loc[hour]):
            sigma_values.append(float(hourly_quantile.loc[hour]))
        else:
            sigma_values.append(epsilon_90_raw)
            if TEST_START <= row["date"] <= TEST_END:
                fallback_count += 1
    df["sigma_u_raw_mwh"] = sigma_values

    df["renewable_scaled_mw"] = df["renewable_mwh_raw"] * renewable_scale_factor
    df["load_scaled_mw"] = df["load_mw_raw"] * load_scale_factor
    df["forecast_renewable_scaled_mw"] = df["forecast_renewable_raw_mwh"] * renewable_scale_factor
    df["sigma_u_scaled_mw"] = df["sigma_u_raw_mwh"] * renewable_scale_factor
    cloud_norm = (df["cloud_cover"] / 100.0).clip(0, 1)
    df["sigma_heuristic_scaled_mw"] = df["forecast_renewable_scaled_mw"] * (0.10 + 0.25 * cloud_norm)

    calib_eval = df.loc[calib_mask].dropna(subset=["forecast_renewable_raw_mwh"])
    test_eval = df.loc[test_mask].dropna(subset=["forecast_renewable_raw_mwh"])
    metrics = {
        "n_train": len(train),
        "n_calibration": len(calib_eval),
        "n_test": len(test_eval),
        "mae_calibration_raw": mean_absolute_error(calib_eval["renewable_mwh_raw"], calib_eval["forecast_renewable_raw_mwh"]),
        "rmse_calibration_raw": float(
            np.sqrt(mean_squared_error(calib_eval["renewable_mwh_raw"], calib_eval["forecast_renewable_raw_mwh"]))
        ),
        "r2_calibration": r2_score(calib_eval["renewable_mwh_raw"], calib_eval["forecast_renewable_raw_mwh"]),
        "mae_test_raw": mean_absolute_error(test_eval["renewable_mwh_raw"], test_eval["forecast_renewable_raw_mwh"]),
        "rmse_test_raw": float(np.sqrt(mean_squared_error(test_eval["renewable_mwh_raw"], test_eval["forecast_renewable_raw_mwh"]))),
        "r2_test": r2_score(test_eval["renewable_mwh_raw"], test_eval["forecast_renewable_raw_mwh"]),
        "epsilon_80_raw": epsilon_80_raw,
        "epsilon_90_raw": epsilon_90_raw,
        "epsilon_95_raw": epsilon_95_raw,
        "epsilon_90_scaled": epsilon_90_raw * renewable_scale_factor,
        "hourly_epsilon_fallback_count": fallback_count,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "calibration_start": CALIB_START,
        "calibration_end": CALIB_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "region": REGION_FIXED,
        "fuel_mode": FUEL_MODE_FIXED,
        "renewable_p95_raw": renewable_p95_raw,
        "load_mean_raw": load_mean_raw,
        "renewable_scale_factor": renewable_scale_factor,
        "load_scale_factor": load_scale_factor,
    }
    hourly_eps = pd.DataFrame(
        {
            "hour": range(24),
            "epsilon_90_by_hour_raw": [float(hourly_quantile.get(h, epsilon_90_raw)) for h in range(24)],
            "calibration_count_by_hour": [int(hourly_count.get(h, 0)) for h in range(24)],
        }
    )
    return df, hourly_eps, metrics


def select_main_valid_dates(modeling: pd.DataFrame, exclusion_rows: list[dict]) -> pd.DataFrame:
    test = modeling[(modeling["datetime"] >= TEST_START) & (modeling["datetime"] <= f"{TEST_END} 23:00:00")].copy()
    required = [
        "renewable_scaled_mw",
        "forecast_renewable_scaled_mw",
        "sigma_u_scaled_mw",
        "sigma_heuristic_scaled_mw",
        "load_scaled_mw",
    ]
    invalid = test[test[required].isna().any(axis=1)]
    if not invalid.empty:
        add_exclusion(exclusion_rows, invalid[["datetime"]], "missing_required_feature", "final_test_selection")
    counts = test.dropna(subset=required).groupby("date")["hour"].nunique().reset_index(name="valid_hour_count")
    main = counts[counts["valid_hour_count"] == 24].copy()
    incomplete = counts[counts["valid_hour_count"] != 24]
    for _, row in incomplete.iterrows():
        exclusion_rows.append(
            {
                "datetime": pd.NaT,
                "date": row["date"],
                "reason": "missing_required_feature",
                "source_stage": "valid_day_selection",
            }
        )
    main["analysis_type"] = "main_all_valid_may"
    main.to_csv(OUTPUT_DATA_DIR / "main_simulation_dates.csv", index=False)
    return main


def select_top_risk_sensitivity_dates(modeling: pd.DataFrame, main_dates: pd.DataFrame) -> pd.DataFrame:
    if main_dates.empty:
        selected = pd.DataFrame(
            columns=["date", "risk_score", "daily_max_renewable_scaled", "daily_mean_load_scaled", "daily_max_ratio", "analysis_type"]
        )
        selected.to_csv(OUTPUT_DATA_DIR / "selected_risk_dates_sensitivity.csv", index=False)
        return selected
    test = modeling[modeling["date"].isin(main_dates["date"])].copy()
    test["surplus_risk"] = test["renewable_scaled_mw"] - 0.86 * test["load_scaled_mw"]
    test["ratio_risk"] = test["renewable_scaled_mw"] / test["load_scaled_mw"].replace(0, np.nan)
    daily = (
        test.groupby("date")
        .agg(
            risk_score=("surplus_risk", "max"),
            daily_max_renewable_scaled=("renewable_scaled_mw", "max"),
            daily_mean_load_scaled=("load_scaled_mw", "mean"),
            daily_max_ratio=("ratio_risk", "max"),
        )
        .reset_index()
    )
    if daily["risk_score"].max() <= 0:
        daily["risk_score"] = daily["daily_max_ratio"]
    selected = daily.sort_values("risk_score", ascending=False).head(10).copy()
    selected["analysis_type"] = "sensitivity_top_risk"
    selected.to_csv(OUTPUT_DATA_DIR / "selected_risk_dates_sensitivity.csv", index=False)
    return selected


def build_real_data_scenarios(
    modeling: pd.DataFrame,
    dates_df: pd.DataFrame,
    analysis_type: str,
) -> list[Scenario]:
    scenarios = []
    for date in dates_df["date"].tolist():
        day = modeling[modeling["date"] == date].sort_values("hour").copy()
        if len(day) != 24:
            continue
        total_load = day["load_scaled_mw"].to_numpy()
        load_by_node = total_load[:, None] * LOAD_WEIGHTS[None, :]
        renewable_share = np.tile(RENEWABLE_WEIGHTS, (TIME_HORIZON, 1))
        scenarios.append(
            Scenario(
                forecast_renewable=day["forecast_renewable_scaled_mw"].to_numpy(),
                actual_renewable=day["renewable_scaled_mw"].to_numpy(),
                sigma_u=day["sigma_u_scaled_mw"].to_numpy(),
                sigma_heuristic=day["sigma_heuristic_scaled_mw"].to_numpy(),
                load_by_node=load_by_node,
                renewable_share_by_node=renewable_share,
                actual_renewable_raw=day["renewable_mwh_raw"].to_numpy(),
                forecast_renewable_raw=day["forecast_renewable_raw_mwh"].to_numpy(),
                sigma_u_raw=day["sigma_u_raw_mwh"].to_numpy(),
                total_load_raw=day["load_mw_raw"].to_numpy(),
                total_load_scaled=total_load,
                date=date,
                subject_id=date,
                analysis_type=analysis_type,
                hourly_frame=day,
            )
        )
    return scenarios


def price_curve() -> np.ndarray:
    hours = np.arange(TIME_HORIZON)
    return 72 + 18 * np.exp(-((hours - 9) / 3.4) ** 2) + 43 * np.exp(-((hours - 19) / 2.8) ** 2)


def make_synthetic_scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    hours = np.arange(TIME_HORIZON)
    solar_shape = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) ** 1.55
    wind_shape = 0.42 + 0.18 * np.sin((hours + seed) / 24 * 2 * np.pi) + rng.normal(0, 0.035, TIME_HORIZON)
    wind_shape = np.clip(wind_shape, 0.12, None)
    raw_renewable = 0.72 * solar_shape + 0.28 * wind_shape
    forecast_renewable = CAP_RENEWABLE * raw_renewable / raw_renewable.max()
    cloud_volatility = np.clip(
        0.5 + 0.35 * np.sin((hours - 9 + seed * 0.4) / 24 * 2 * np.pi) + rng.normal(0, 0.17, TIME_HORIZON),
        0,
        1,
    )
    sigma_u = forecast_renewable * (0.10 + 0.25 * cloud_volatility)
    actual_renewable = np.clip(forecast_renewable + rng.normal(0, sigma_u), 0, CAP_RENEWABLE * 1.12)
    total_load = (
        145
        + 36 * np.exp(-((hours - 8) / 3.2) ** 2)
        + 75 * np.exp(-((hours - 19) / 3.0) ** 2)
        + rng.normal(0, 6, TIME_HORIZON)
    )
    total_load = np.clip(total_load, 115, 245)
    node_noise = rng.normal(1.0, 0.05, (TIME_HORIZON, N_NODES))
    node_noise = np.clip(node_noise, 0.85, 1.18)
    load_by_node = total_load[:, None] * LOAD_WEIGHTS[None, :] * node_noise
    load_by_node *= total_load[:, None] / load_by_node.sum(axis=1, keepdims=True)
    renewable_noise = rng.normal(1.0, 0.05, (TIME_HORIZON, len(RENEWABLE_NODES)))
    renewable_share_by_node = RENEWABLE_WEIGHTS[None, :] * renewable_noise
    renewable_share_by_node = np.clip(renewable_share_by_node, 0.03, None)
    renewable_share_by_node /= renewable_share_by_node.sum(axis=1, keepdims=True)
    return Scenario(
        forecast_renewable=forecast_renewable,
        actual_renewable=actual_renewable,
        sigma_u=sigma_u,
        sigma_heuristic=sigma_u,
        load_by_node=load_by_node,
        renewable_share_by_node=renewable_share_by_node,
        actual_renewable_raw=actual_renewable,
        forecast_renewable_raw=forecast_renewable,
        sigma_u_raw=sigma_u,
        total_load_raw=total_load,
        total_load_scaled=total_load,
        subject_id=str(seed),
        date=f"synthetic_seed_{seed}",
        analysis_type="synthetic",
    )


def make_injection(
    renewable_total: float,
    renewable_share: np.ndarray,
    load_nodes: np.ndarray,
    ess_action_nodes: np.ndarray,
) -> np.ndarray:
    injection = -load_nodes.copy()
    for pos, node in enumerate(RENEWABLE_NODES):
        injection[node] += renewable_total * renewable_share[pos]
    for pos, node in enumerate(ESS_NODES):
        injection[node] += ess_action_nodes[pos]
    return injection


def forecast_line_stress(scenario: Scenario, hour: int, renewable_value: float | None = None) -> float:
    renewable = scenario.forecast_renewable[hour] if renewable_value is None else renewable_value
    injection = make_injection(
        renewable,
        scenario.renewable_share_by_node[hour],
        scenario.load_by_node[hour],
        np.zeros(len(ESS_NODES)),
    )
    line_stress = np.max(np.abs(dc_power_flow(injection)))
    local_stress = np.max(injection[RENEWABLE_NODES]) / LOCAL_RENEWABLE_EXPORT_LIMIT * LINE_TRIP_LIMIT
    return max(line_stress, local_stress)


def deterministic_plan(scenario: Scenario) -> np.ndarray:
    plan = np.zeros((TIME_HORIZON, len(ESS_NODES)))
    soc = np.full(len(ESS_NODES), ESS_NODE_CAP * 0.45)
    price = price_curve()
    for hour in range(TIME_HORIZON):
        net_forecast = scenario.forecast_renewable[hour] - scenario.load_by_node[hour].sum()
        action_total = 0.0
        if net_forecast > 0 or 10 <= hour <= 15:
            action_total = -min(ESS_NODE_POWER * len(ESS_NODES), max(35.0, net_forecast * 0.65 + 70.0))
        elif price[hour] > np.percentile(price, 65):
            action_total = min(ESS_NODE_POWER * len(ESS_NODES), 95.0)
        per_node = action_total / len(ESS_NODES)
        for i in range(len(ESS_NODES)):
            if per_node < 0:
                feasible = min(-per_node, (ESS_NODE_CAP - soc[i]) / ESS_EFF, ESS_NODE_POWER)
                plan[hour, i] = -feasible
                soc[i] += feasible * ESS_EFF
            else:
                feasible = min(per_node, soc[i] * ESS_EFF, ESS_NODE_POWER)
                plan[hour, i] = feasible
                soc[i] -= feasible / ESS_EFF
    return plan


def dispatch_action(mode: str, scenario: Scenario, hour: int, soc: np.ndarray, det_plan: np.ndarray) -> tuple[np.ndarray, float]:
    total_load = scenario.load_by_node[hour].sum()
    forecast = scenario.forecast_renewable[hour]
    actual = scenario.actual_renewable[hour]
    sigma_mode = scenario.sigma_u[hour]
    if mode == "Stochastic_Proposed" and scenario.sigma_heuristic is not None:
        sigma_mode = scenario.sigma_heuristic[hour]
    price = price_curve()[hour]

    action = np.zeros(len(ESS_NODES))
    risk_renewable = forecast + STOCHASTIC_RISK_Z * sigma_mode
    predicted_stress = forecast_line_stress(scenario, hour, risk_renewable)
    risk_score = float(np.exp(min(6.0, predicted_stress / LINE_TRIP_LIMIT)))

    if mode == "Heuristic_Rule":
        if 11 <= hour <= 14:
            action[:] = -42.0
        elif 18 <= hour <= 21:
            action[:] = 45.0
    elif mode == "Deterministic_Opt":
        action = det_plan[hour].copy()
    elif mode == "Rolling_Greedy":
        trial = make_injection(actual, scenario.renewable_share_by_node[hour], scenario.load_by_node[hour], action)
        overload = max(0.0, np.max(np.abs(dc_power_flow(trial))) - LINE_TRIP_LIMIT)
        local_excess = np.clip(trial[RENEWABLE_NODES] - LOCAL_RENEWABLE_EXPORT_LIMIT, 0, None)
        surplus = max(0.0, actual - total_load * 0.86)
        deficit = max(0.0, total_load * 0.92 - actual)
        if overload > 0 or surplus > 0 or local_excess.sum() > 0:
            action = -np.minimum(ESS_NODE_POWER, local_excess + (overload * 2.3 + surplus * 0.45) / len(ESS_NODES))
        elif deficit > 0 and price > 82:
            action[:] = min(ESS_NODE_POWER, deficit * 0.32 / len(ESS_NODES))
    elif mode == "Stochastic_Proposed":
        action = stochastic_objective_dispatch(scenario, hour, soc, sigma_override=scenario.sigma_heuristic)
    elif mode == "SVR_ResidualQuantile":
        action = stochastic_objective_dispatch(scenario, hour, soc, sigma_override=scenario.sigma_u)

    return apply_ess_limits(action, soc), risk_score


def stochastic_objective_dispatch(
    scenario: Scenario,
    hour: int,
    soc: np.ndarray,
    sigma_override: np.ndarray | None = None,
) -> np.ndarray:
    local_trial = make_injection(
        scenario.actual_renewable[hour],
        scenario.renewable_share_by_node[hour],
        scenario.load_by_node[hour],
        np.zeros(len(ESS_NODES)),
    )
    local_excess = np.clip(local_trial[RENEWABLE_NODES] - LOCAL_RENEWABLE_EXPORT_LIMIT * 0.88, 0, None)
    future_need = forecast_absorption_need(scenario, hour, sigma_override=sigma_override)
    current_room = ESS_NODE_CAP - soc
    reserve_shortage_now = np.clip(future_need - current_room, 0, None)
    price = price_curve()[hour]

    absorb_action = -np.minimum(ESS_NODE_POWER, local_excess)
    reserve_action = np.minimum(ESS_NODE_POWER, reserve_shortage_now * ESS_EFF)
    mixed_action = np.where(local_excess > 0, absorb_action, reserve_action)
    economic_action = np.minimum(ESS_NODE_POWER * 0.45, soc * ESS_EFF * 0.18) if price > 100 else np.zeros(len(ESS_NODES))
    candidates = [np.zeros(len(ESS_NODES)), absorb_action, reserve_action, mixed_action, economic_action]

    best_score = float("inf")
    best_action = np.zeros(len(ESS_NODES))
    for requested in candidates:
        action = apply_ess_limits(requested, soc)
        next_soc = update_soc(soc, action)
        curtailment, flows_after = curtailment_for_renewable(
            scenario.actual_renewable[hour],
            scenario.renewable_share_by_node[hour],
            scenario.load_by_node[hour],
            action,
        )
        injection_after = make_injection(
            scenario.actual_renewable[hour] - curtailment,
            scenario.renewable_share_by_node[hour],
            scenario.load_by_node[hour],
            action,
        )
        local_export_after = np.clip(injection_after[RENEWABLE_NODES], 0, None)
        line_over = max(0.0, np.max(np.abs(flows_after)) - LINE_TRIP_LIMIT)
        local_over = np.clip(local_export_after - LOCAL_RENEWABLE_EXPORT_LIMIT, 0, None).sum()
        room_after = ESS_NODE_CAP - next_soc
        reserve_shortage = np.clip(future_need - room_after, 0, None).sum()
        missed_current_absorption = np.clip(local_excess + action, 0, None).sum()
        throughput = np.abs(action).sum()
        score = (
            10.0 * curtailment
            + 16.0 * local_over
            + 11.0 * line_over
            + 4.5 * reserve_shortage
            + 2.0 * missed_current_absorption
            + 0.18 * throughput
        )
        if score < best_score:
            best_score = score
            best_action = action
    return best_action


def forecast_absorption_need(scenario: Scenario, hour: int, sigma_override: np.ndarray | None = None) -> np.ndarray:
    need = np.zeros(len(ESS_NODES))
    sigma_series = scenario.sigma_u if sigma_override is None else sigma_override
    for h in range(hour + 1, min(TIME_HORIZON, hour + 1 + STOCHASTIC_LOOKAHEAD)):
        risk_renewable = scenario.forecast_renewable[h] + STOCHASTIC_RISK_Z * sigma_series[h]
        for pos, node in enumerate(RENEWABLE_NODES):
            local_generation = risk_renewable * scenario.renewable_share_by_node[h, pos]
            local_load = scenario.load_by_node[h, node]
            local_export = local_generation - local_load
            need[pos] += max(0.0, local_export - LOCAL_RENEWABLE_EXPORT_LIMIT * 0.82)
    return np.clip(need, 0, ESS_NODE_CAP)


def apply_ess_limits(action: np.ndarray, soc: np.ndarray) -> np.ndarray:
    limited = np.zeros_like(action)
    for i, requested in enumerate(action):
        if requested < 0:
            feasible_charge = min(-requested, (ESS_NODE_CAP - soc[i]) / ESS_EFF, ESS_NODE_POWER)
            limited[i] = -max(0.0, feasible_charge)
        else:
            feasible_discharge = min(requested, soc[i] * ESS_EFF, ESS_NODE_POWER)
            limited[i] = max(0.0, feasible_discharge)
    return limited


def update_soc(soc: np.ndarray, action: np.ndarray) -> np.ndarray:
    next_soc = soc.copy()
    for i, value in enumerate(action):
        if value < 0:
            next_soc[i] += (-value) * ESS_EFF
        else:
            next_soc[i] -= value / ESS_EFF
    return np.clip(next_soc, 0, ESS_NODE_CAP)


def curtailment_for_renewable(
    renewable_total: float,
    renewable_share: np.ndarray,
    load_nodes: np.ndarray,
    ess_action: np.ndarray,
) -> tuple[float, np.ndarray]:
    injection = make_injection(renewable_total, renewable_share, load_nodes, ess_action)
    flows = dc_power_flow(injection)
    renewable_exports = injection[RENEWABLE_NODES]
    original_stress = max(np.max(np.abs(flows)), np.max(renewable_exports) / LOCAL_RENEWABLE_EXPORT_LIMIT * LINE_TRIP_LIMIT)
    if np.max(np.abs(flows)) <= LINE_TRIP_LIMIT and np.max(renewable_exports) <= LOCAL_RENEWABLE_EXPORT_LIMIT:
        return 0.0, flows

    candidates = np.linspace(0.0, renewable_total, 241)
    stresses = []
    flow_candidates = []
    for curtailment in candidates:
        trial_renewable = renewable_total - curtailment
        trial_injection = make_injection(trial_renewable, renewable_share, load_nodes, ess_action)
        trial_flows = dc_power_flow(trial_injection)
        trial_exports = trial_injection[RENEWABLE_NODES]
        stress = max(np.max(np.abs(trial_flows)), np.max(trial_exports) / LOCAL_RENEWABLE_EXPORT_LIMIT * LINE_TRIP_LIMIT)
        stresses.append(stress)
        flow_candidates.append(trial_flows)
    stresses = np.array(stresses)
    safe_idx = np.where(stresses <= LINE_TRIP_LIMIT)[0]
    if len(safe_idx) > 0:
        idx = int(safe_idx[0])
        return float(candidates[idx]), flow_candidates[idx]
    best_idx = int(np.argmin(stresses))
    if stresses[best_idx] < original_stress:
        return float(candidates[best_idx]), flow_candidates[best_idx]
    return 0.0, flows


def simulate_mode(subject_id: str, mode: str, scenario: Scenario) -> tuple[list[dict], list[dict], dict]:
    soc = np.full(len(ESS_NODES), ESS_NODE_CAP * 0.45)
    det_plan = deterministic_plan(scenario)
    rows = []
    line_rows = []
    total_curtailment = 0.0
    total_unserved = 0.0
    throughput = 0.0
    total_charge = 0.0
    total_discharge = 0.0
    soc_values = []
    max_line_loading_values = []
    mean_line_loading_values = []

    for hour in range(TIME_HORIZON):
        action, risk_score = dispatch_action(mode, scenario, hour, soc, det_plan)
        before_injection = make_injection(
            scenario.actual_renewable[hour],
            scenario.renewable_share_by_node[hour],
            scenario.load_by_node[hour],
            action,
        )
        flows_before = dc_power_flow(before_injection)
        curtailment, flows_after = curtailment_for_renewable(
            scenario.actual_renewable[hour],
            scenario.renewable_share_by_node[hour],
            scenario.load_by_node[hour],
            action,
        )
        renewable_used = scenario.actual_renewable[hour] - curtailment
        discharge = np.clip(action, 0, None).sum()
        charge = np.clip(-action, 0, None).sum()
        total_load = scenario.load_by_node[hour].sum()
        unserved = max(0.0, total_load - renewable_used - discharge - 0.18 * total_load)

        total_curtailment += curtailment * DT
        total_unserved += unserved * DT
        total_charge += charge * DT
        total_discharge += discharge * DT
        throughput += (charge + discharge) * DT
        soc = update_soc(soc, action)
        soc_values.append(float(soc.sum()))

        utilization = np.abs(flows_after) / LINE_TRIP_LIMIT
        max_line_loading_values.append(float(utilization.max()))
        mean_line_loading_values.append(float(utilization.mean()))
        line_overload_count = int((utilization > 1.0).sum())

        rows.append(
            {
                "subject_id": subject_id,
                "seed": subject_id if scenario.analysis_type == "synthetic" else np.nan,
                "date": scenario.date,
                "hour": hour,
                "mode": mode,
                "analysis_type": scenario.analysis_type,
                "actual_renewable_raw_mwh": scenario.actual_renewable_raw[hour],
                "forecast_renewable_raw_mwh": scenario.forecast_renewable_raw[hour],
                "sigma_u_raw_mwh": scenario.sigma_u_raw[hour],
                "actual_renewable_scaled_mw": scenario.actual_renewable[hour],
                "forecast_renewable_scaled_mw": scenario.forecast_renewable[hour],
                "sigma_u_scaled_mw": scenario.sigma_u[hour],
                "sigma_heuristic_scaled_mw": scenario.sigma_heuristic[hour] if scenario.sigma_heuristic is not None else scenario.sigma_u[hour],
                "total_load_raw_mw": scenario.total_load_raw[hour],
                "total_load_scaled_mw": scenario.total_load_scaled[hour],
                "forecast_renewable_mw": scenario.forecast_renewable[hour],
                "actual_renewable_mw": scenario.actual_renewable[hour],
                "sigma_u_mw": scenario.sigma_u[hour],
                "total_load_mw": total_load,
                "ess_charge_mw": charge,
                "ess_discharge_mw": discharge,
                "ess_soc_mwh": soc.sum(),
                "soc_total_mwh": soc.sum(),
                "soc_percent": soc.sum() / ESS_TOTAL_CAP * 100,
                "curtailment_mw": curtailment,
                "curtailment_mwh": curtailment * DT,
                "unserved_energy_mw": unserved,
                "risk_score": risk_score,
                "line_overload_count": line_overload_count,
                "max_line_flow_before_mw": np.max(np.abs(flows_before)),
                "max_line_flow_after_mw": np.max(np.abs(flows_after)),
                "max_line_loading": utilization.max(),
                "mean_line_loading": utilization.mean(),
            }
        )
        for line_id, flow in enumerate(flows_after, start=1):
            line_rows.append(
                {
                    "subject_id": subject_id,
                    "seed": subject_id if scenario.analysis_type == "synthetic" else np.nan,
                    "date": scenario.date,
                    "mode": mode,
                    "analysis_type": scenario.analysis_type,
                    "hour": hour,
                    "line_id": line_id,
                    "flow_mw": flow,
                    "abs_flow_mw": abs(flow),
                    "utilization_pct": abs(flow) / LINE_TRIP_LIMIT * 100,
                    "line_loading": abs(flow) / LINE_TRIP_LIMIT,
                }
            )

    summary = {
        "subject_id": subject_id,
        "seed": subject_id if scenario.analysis_type == "synthetic" else np.nan,
        "date": scenario.date,
        "mode": mode,
        "analysis_type": scenario.analysis_type,
        "total_curtailment_mwh": total_curtailment,
        "daily_curtailment_mwh": total_curtailment,
        "total_unserved_mwh": total_unserved,
        "max_line_loading": max(max_line_loading_values),
        "mean_line_loading": float(np.mean(mean_line_loading_values)),
        "final_soc_mwh": soc_values[-1],
        "min_soc_mwh": min(soc_values),
        "max_soc_mwh": max(soc_values),
        "total_charge_mwh": total_charge,
        "total_discharge_mwh": total_discharge,
        "ess_utilization_pct": throughput / ESS_TOTAL_CAP * 100,
    }
    return rows, line_rows, summary


def repeated_measures_anova(summary_df: pd.DataFrame, modes: list[str], analysis_type: str) -> pd.DataFrame:
    subset = summary_df[summary_df["analysis_type"] == analysis_type]
    wide = subset.pivot(index="subject_id", columns="mode", values="total_curtailment_mwh").reindex(columns=modes).dropna()
    if len(wide) < 2:
        return pd.DataFrame(
            [
                {
                    "analysis_type": analysis_type,
                    "metric": "total_curtailment_mwh",
                    "factor": "mode",
                    "f_statistic": np.nan,
                    "p_value": np.nan,
                    "note": "Not enough repeated-measures subjects.",
                }
            ]
        )
    values = wide.to_numpy()
    n_subjects, n_conditions = values.shape
    grand_mean = values.mean()
    condition_means = values.mean(axis=0)
    subject_means = values.mean(axis=1)
    ss_total = ((values - grand_mean) ** 2).sum()
    ss_conditions = n_subjects * ((condition_means - grand_mean) ** 2).sum()
    ss_subjects = n_conditions * ((subject_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_conditions - ss_subjects
    df_conditions = n_conditions - 1
    df_subjects = n_subjects - 1
    df_error = df_conditions * df_subjects
    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error if df_error else np.nan
    f_stat = ms_conditions / ms_error if ms_error and ms_error > 0 else np.nan
    p_value = stats.f.sf(f_stat, df_conditions, df_error) if pd.notna(f_stat) else np.nan
    partial_eta_squared = ss_conditions / (ss_conditions + ss_error) if (ss_conditions + ss_error) else np.nan
    return pd.DataFrame(
        [
            {
                "analysis_type": analysis_type,
                "metric": "total_curtailment_mwh",
                "factor": "mode",
                "ss_factor": ss_conditions,
                "df_factor": df_conditions,
                "ms_factor": ms_conditions,
                "ss_subject": ss_subjects,
                "df_subject": df_subjects,
                "ss_error": ss_error,
                "df_error": df_error,
                "ms_error": ms_error,
                "f_statistic": f_stat,
                "p_value": p_value,
                "partial_eta_squared": partial_eta_squared,
                "note": "Repeated-measures ANOVA with date as the subject.",
            }
        ]
    )


def paired_tests(summary_df: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
    subset = summary_df[summary_df["analysis_type"] == analysis_type]
    wide = subset.pivot(index="subject_id", columns="mode", values="total_curtailment_mwh")
    rows = []
    for proposed in ["Stochastic_Proposed", "SVR_ResidualQuantile"]:
        if proposed not in wide.columns:
            continue
        for baseline in ["Heuristic_Rule", "Deterministic_Opt", "Rolling_Greedy"]:
            if baseline not in wide.columns:
                continue
            paired = wide[[baseline, proposed]].dropna()
            if len(paired) >= 2:
                t_result = stats.ttest_rel(paired[baseline], paired[proposed])
                try:
                    w_result = stats.wilcoxon(paired[baseline], paired[proposed])
                    w_stat, w_p = w_result.statistic, w_result.pvalue
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
            else:
                t_result = stats.ttest_rel([np.nan], [np.nan])
                w_stat, w_p = np.nan, np.nan
            diff = paired[baseline] - paired[proposed]
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "metric": "total_curtailment_mwh",
                    "baseline_mode": baseline,
                    "proposed_mode": proposed,
                    "mean_baseline_minus_proposed_mwh": diff.mean() if not diff.empty else np.nan,
                    "paired_t_statistic": t_result.statistic,
                    "paired_t_p_value": t_result.pvalue,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_value": w_p,
                    "n_paired_subjects": len(paired),
                    "note": "Paired comparison by identical date subject.",
                }
            )
    return pd.DataFrame(rows)


def run_scenarios(scenarios: list[Scenario], modes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_rows = []
    line_rows = []
    summary_rows = []
    for scenario in scenarios:
        subject_id = scenario.subject_id or scenario.date or "unknown"
        for mode in modes:
            t_rows, l_rows, summary = simulate_mode(subject_id, mode, scenario)
            time_rows.extend(t_rows)
            line_rows.extend(l_rows)
            summary_rows.append(summary)
    return pd.DataFrame(time_rows), pd.DataFrame(line_rows), pd.DataFrame(summary_rows)


def run_real_data_simulation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    require_real_data_files()
    exclusion_rows: list[dict] = []
    demand = read_demand_files()
    renewable = read_renewable_file(exclusion_rows)
    weather = read_weather_file()
    modeling = build_modeling_table(renewable, demand, weather, exclusion_rows)
    modeling, hourly_eps, metrics = train_svr_forecaster(modeling)

    main_dates = select_main_valid_dates(modeling, exclusion_rows)
    risk_dates = select_top_risk_sensitivity_dates(modeling, main_dates)
    main_scenarios = build_real_data_scenarios(modeling, main_dates, "main_all_valid_may")
    sensitivity_scenarios = build_real_data_scenarios(modeling, risk_dates, "sensitivity_top_risk")

    time_main, line_main, summary_main = run_scenarios(main_scenarios, REAL_MODES)
    time_sens, line_sens, summary_sens = run_scenarios(sensitivity_scenarios, REAL_MODES)
    time_df = pd.concat([time_main, time_sens], ignore_index=True)
    line_df = pd.concat([line_main, line_sens], ignore_index=True)
    summary_df = pd.concat([summary_main, summary_sens], ignore_index=True)

    rm_df = pd.concat(
        [
            repeated_measures_anova(summary_df, REAL_MODES, "main_all_valid_may"),
            repeated_measures_anova(summary_df, REAL_MODES, "sensitivity_top_risk"),
        ],
        ignore_index=True,
    )
    paired_df = pd.concat(
        [
            paired_tests(summary_df, "main_all_valid_may"),
            paired_tests(summary_df, "sensitivity_top_risk"),
        ],
        ignore_index=True,
    )
    groups = [
        summary_df[(summary_df["analysis_type"] == "main_all_valid_may") & (summary_df["mode"] == mode)][
            "total_curtailment_mwh"
        ].values
        for mode in REAL_MODES
    ]
    if all(len(g) > 1 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = np.nan, np.nan
    anova_df = pd.DataFrame(
        [
            {
                "analysis_type": "main_all_valid_may",
                "metric": "total_curtailment_mwh",
                "f_statistic": f_stat,
                "p_value": p_value,
                "note": "Independent one-way ANOVA retained for backward compatibility; main inference uses repeated measures.",
            }
        ]
    )

    modeling.to_csv(OUTPUT_DATA_DIR / "modeling_table.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DATA_DIR / "svr_metrics.csv", index=False)
    hourly_eps.to_csv(OUTPUT_DATA_DIR / "svr_hourly_epsilon.csv", index=False)
    save_exclusion_log(exclusion_rows)
    save_assumptions(metrics)
    save_data_warnings(exclusion_rows, main_dates, metrics)

    time_df.to_csv(OUTPUT_DATA_DIR / "time_series.csv", index=False)
    line_df.to_csv(OUTPUT_DATA_DIR / "line_flows.csv", index=False)
    summary_df.to_csv(OUTPUT_DATA_DIR / "simulation_summary.csv", index=False)
    anova_df.to_csv(OUTPUT_DATA_DIR / "anova_result.csv", index=False)
    rm_df.to_csv(OUTPUT_DATA_DIR / "repeated_measures_anova.csv", index=False)
    paired_df.to_csv(OUTPUT_DATA_DIR / "paired_tests.csv", index=False)
    plot_figures(time_df, line_df, summary_df, modeling, main_dates, risk_dates)
    return time_df, line_df, summary_df, anova_df, rm_df, paired_df


def run_synthetic_simulation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    scenarios = [make_synthetic_scenario(seed) for seed in range(1, N_SEEDS + 1)]
    time_df, line_df, summary_df = run_scenarios(scenarios, SYNTHETIC_MODES)
    rm_df = repeated_measures_anova(summary_df, SYNTHETIC_MODES, "synthetic")
    paired_df = paired_tests(summary_df, "synthetic")
    groups = [summary_df[summary_df["mode"] == mode]["total_curtailment_mwh"].values for mode in SYNTHETIC_MODES]
    f_stat, p_value = stats.f_oneway(*groups)
    anova_df = pd.DataFrame(
        [
            {
                "analysis_type": "synthetic",
                "metric": "total_curtailment_mwh",
                "f_statistic": f_stat,
                "p_value": p_value,
                "note": "Synthetic fallback independent one-way ANOVA.",
            }
        ]
    )
    time_df.to_csv(OUTPUT_DATA_DIR / "time_series.csv", index=False)
    line_df.to_csv(OUTPUT_DATA_DIR / "line_flows.csv", index=False)
    summary_df.to_csv(OUTPUT_DATA_DIR / "simulation_summary.csv", index=False)
    anova_df.to_csv(OUTPUT_DATA_DIR / "anova_result.csv", index=False)
    rm_df.to_csv(OUTPUT_DATA_DIR / "repeated_measures_anova.csv", index=False)
    paired_df.to_csv(OUTPUT_DATA_DIR / "paired_tests.csv", index=False)
    plot_synthetic_figures(time_df, line_df, summary_df)
    return time_df, line_df, summary_df, anova_df, rm_df, paired_df


def save_exclusion_log(exclusion_rows: list[dict]) -> None:
    columns = ["datetime", "date", "reason", "source_stage"]
    df = pd.DataFrame(exclusion_rows, columns=columns)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.to_csv(OUTPUT_DATA_DIR / "exclusion_log.csv", index=False)


def save_assumptions(metrics: dict) -> None:
    assumptions = {
        "region": REGION_FIXED,
        "fuel_mode": FUEL_MODE_FIXED,
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "calibration_start": CALIB_START,
        "calibration_end": CALIB_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "cap_renewable": CAP_RENEWABLE,
        "load_target_mean": LOAD_TARGET_MEAN,
        "ess_total_cap": ESS_TOTAL_CAP,
        "ess_node_power": ESS_NODE_POWER,
        "line_trip_limit": LINE_TRIP_LIMIT,
        "local_renewable_export_limit": LOCAL_RENEWABLE_EXPORT_LIMIT,
        "stochastic_risk_z": STOCHASTIC_RISK_Z,
        "svr_model": {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"},
        "features": SVR_FEATURES,
        "renewable_p95_raw_training_only": metrics["renewable_p95_raw"],
        "load_mean_raw_training_only": metrics["load_mean_raw"],
        "renewable_scale_factor": metrics["renewable_scale_factor"],
        "load_scale_factor": metrics["load_scale_factor"],
        "integrity_note": "Constants and periods are fixed before algorithm comparison; May test data is not used for uncertainty calibration.",
    }
    with (OUTPUT_DATA_DIR / "assumptions.json").open("w", encoding="utf-8") as handle:
        json.dump(assumptions, handle, ensure_ascii=False, indent=2)


def save_data_warnings(exclusion_rows: list[dict], main_dates: pd.DataFrame, metrics: dict) -> None:
    expected_may_days = pd.date_range(TEST_START, TEST_END, freq="D").strftime("%Y-%m-%d").tolist()
    valid_dates = set(main_dates["date"].tolist())
    excluded_dates = sorted(set(expected_may_days) - valid_dates)
    lines = [
        "Data limitations:",
        "- The 14-bus / 20-line DC network is a simplified toy grid and is not the actual Korean transmission grid.",
        "- KPX raw renewable and demand values are scaled into the toy VPP grid using fixed training-period scaling factors.",
        "- Open-Meteo weather is not plant-site measured sensor data.",
        "- KPX market/transaction data may not represent every distributed generator.",
        f"Excluded May dates count: {len(excluded_dates)}",
        f"Excluded row/date log entries count: {len(exclusion_rows)}",
        f"Final test dates are all valid May dates: {len(excluded_dates) == 0}",
        "Top-risk analysis is sensitivity-only and is not the basis for the main conclusion.",
        "Mainland curtailment files are not used as direct validation because the simulation uses Jeju weather and Jeju renewable generation.",
        f"Hourly epsilon fallback count in test/simulation rows: {metrics['hourly_epsilon_fallback_count']}",
    ]
    if len(main_dates) < 5:
        lines.append("Main valid simulation dates are fewer than 5, so statistical tests have low reliability.")
    if excluded_dates:
        lines.append("Excluded May dates: " + ", ".join(excluded_dates))
    (OUTPUT_DATA_DIR / "data_warnings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def representative_date(modeling: pd.DataFrame, main_dates: pd.DataFrame) -> str | None:
    if main_dates.empty:
        return None
    test = modeling[modeling["date"].isin(main_dates["date"])].copy()
    test["surplus_risk"] = test["renewable_scaled_mw"] - 0.86 * test["load_scaled_mw"]
    daily = test.groupby("date")["surplus_risk"].max().sort_values()
    return daily.index[len(daily) // 2]


def plot_figures(
    time_df: pd.DataFrame,
    line_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    modeling: pd.DataFrame,
    main_dates: pd.DataFrame,
    risk_dates: pd.DataFrame,
) -> None:
    configure_korean_font()
    colors = {
        "Heuristic_Rule": "#4b5563",
        "Deterministic_Opt": "#2563eb",
        "Rolling_Greedy": "#f97316",
        "Stochastic_Proposed": "#16a34a",
        "SVR_ResidualQuantile": "#dc2626",
    }
    labels = {
        "Heuristic_Rule": "규칙 기반",
        "Deterministic_Opt": "확정 계획",
        "Rolling_Greedy": "실시간 탐욕",
        "Stochastic_Proposed": "구름량 경험 불확실성",
        "SVR_ResidualQuantile": "SVR 잔차 분위수",
    }
    main_summary = summary_df[summary_df["analysis_type"] == "main_all_valid_may"]
    stats_df = (
        main_summary.groupby("mode")["total_curtailment_mwh"].agg(["mean", "std"]).reindex(REAL_MODES).reset_index()
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(REAL_MODES))
    ax.bar(x, stats_df["mean"], yerr=stats_df["std"], capsize=5, color=[colors[m] for m in REAL_MODES])
    ax.set_xticks(x, [labels[m] for m in REAL_MODES], rotation=15, ha="right")
    ax.set_ylabel("총 출력제어량 (MWh/day, scaled)")
    ax.set_title("Main analysis: all valid May 2025 days")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_curtailment_bar.png", dpi=220)
    plt.close(fig)

    rep_date = representative_date(modeling, main_dates)
    if rep_date is None:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for mode in REAL_MODES:
        subset = time_df[
            (time_df["analysis_type"] == "main_all_valid_may") & (time_df["date"] == rep_date) & (time_df["mode"] == mode)
        ]
        ax.plot(subset["hour"], subset["ess_soc_mwh"], marker="o", linewidth=1.8, color=colors[mode], label=labels[mode])
    ax.set_xlabel("시간")
    ax.set_ylabel("ESS SOC (MWh)")
    ax.set_title(f"대표일 ESS SOC 변화: {rep_date} (median-risk)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_soc_timeseries.png", dpi=220)
    plt.close(fig)

    diag = modeling[modeling["date"] == rep_date].sort_values("hour")
    hours = diag["hour"].to_numpy()
    forecast = diag["forecast_renewable_scaled_mw"].to_numpy()
    heuristic = diag["sigma_heuristic_scaled_mw"].to_numpy()
    residual = diag["sigma_u_scaled_mw"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(hours, diag["renewable_scaled_mw"], color="#111827", linewidth=2, label="실제 재생에너지")
    ax.plot(hours, forecast, color="#2563eb", linewidth=2, label="SVR 예측")
    ax.fill_between(hours, forecast - heuristic, forecast + heuristic, color="#16a34a", alpha=0.20, label="구름량 기반 밴드")
    ax.fill_between(hours, forecast - residual, forecast + residual, color="#dc2626", alpha=0.18, label="SVR 잔차 분위수 밴드")
    ax.set_xlabel("시간")
    ax.set_ylabel("Scaled MW")
    ax.set_title(f"불확실성 방식 비교와 재생에너지 리스크: {rep_date}")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_uncertainty_risk.png", dpi=220)
    plt.close(fig)

    heat = (
        line_df[
            (line_df["analysis_type"] == "main_all_valid_may")
            & (line_df["date"] == rep_date)
            & (line_df["mode"] == "SVR_ResidualQuantile")
        ]
        .pivot(index="line_id", columns="hour", values="utilization_pct")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=120)
    ax.set_xticks(range(0, 24, 2), range(0, 24, 2))
    ax.set_yticks(range(0, len(heat.index), 2), heat.index[::2])
    ax.set_xlabel("시간")
    ax.set_ylabel("송전선로 번호")
    ax.set_title(f"SVR 잔차 분위수 모드 선로 이용률: {rep_date}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("선로 이용률 (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_line_heatmap.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(hours, diag["renewable_scaled_mw"], color="#111827", linewidth=2, label="실제")
    ax.plot(hours, forecast, color="#2563eb", linewidth=2, label="SVR 예측")
    ax.fill_between(hours, forecast - residual, forecast + residual, color="#93c5fd", alpha=0.35, label="예측 +/- hourly epsilon90")
    ax.set_xlabel("시간")
    ax.set_ylabel("Scaled MW")
    ax.set_title(f"SVR forecast interval: {rep_date}")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_svr_forecast_interval.png", dpi=220)
    plt.close(fig)

    test = modeling[modeling["date"].isin(main_dates["date"])].copy()
    test["surplus_risk"] = test["renewable_scaled_mw"] - 0.86 * test["load_scaled_mw"]
    test["ratio_risk"] = test["renewable_scaled_mw"] / test["load_scaled_mw"].replace(0, np.nan)
    daily = (
        test.groupby("date")
        .agg(risk_score=("surplus_risk", "max"), daily_max_ratio=("ratio_risk", "max"))
        .reset_index()
        .sort_values("date")
    )
    top_set = set(risk_dates["date"].tolist())
    fig, ax = plt.subplots(figsize=(11, 5.2))
    normal = daily[~daily["date"].isin(top_set)]
    top = daily[daily["date"].isin(top_set)]
    ax.scatter(normal["date"], normal["risk_score"], color="#2563eb", label="Main valid May dates", s=32)
    ax.scatter(top["date"], top["risk_score"], color="#dc2626", label="Top-risk sensitivity dates", s=46)
    ax.set_xlabel("날짜")
    ax.set_ylabel("일 최대 surplus risk")
    ax.set_title("Daily surplus risk: main analysis uses all valid May dates")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure6_daily_surplus_risk.png", dpi=220)
    plt.close(fig)


def plot_synthetic_figures(time_df: pd.DataFrame, line_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    configure_korean_font()
    colors = {
        "Heuristic_Rule": "#4b5563",
        "Deterministic_Opt": "#2563eb",
        "Rolling_Greedy": "#f97316",
        "Stochastic_Proposed": "#16a34a",
    }
    labels = {
        "Heuristic_Rule": "규칙 기반",
        "Deterministic_Opt": "확정 계획",
        "Rolling_Greedy": "실시간 탐욕",
        "Stochastic_Proposed": "확률 리스크 인지",
    }
    stats_df = summary_df.groupby("mode")["total_curtailment_mwh"].agg(["mean", "std"]).reindex(SYNTHETIC_MODES).reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(SYNTHETIC_MODES))
    ax.bar(x, stats_df["mean"], yerr=stats_df["std"], capsize=6, color=[colors[m] for m in SYNTHETIC_MODES])
    ax.set_xticks(x, [labels[m] for m in SYNTHETIC_MODES])
    ax.set_ylabel("일평균 출력 제한량 (MWh)")
    ax.set_title("Synthetic fallback algorithm comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_curtailment_bar.png", dpi=220)
    plt.close(fig)

    seed = "1"
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for mode in SYNTHETIC_MODES:
        subset = time_df[(time_df["subject_id"].astype(str) == seed) & (time_df["mode"] == mode)]
        ax.plot(subset["hour"], subset["soc_percent"], marker="o", linewidth=2, color=colors[mode], label=labels[mode])
    ax.set_xlabel("시간")
    ax.set_ylabel("ESS SOC (%)")
    ax.set_title("Synthetic fallback ESS SOC")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_soc_timeseries.png", dpi=220)
    plt.close(fig)

    diag = time_df[(time_df["subject_id"].astype(str) == seed) & (time_df["mode"] == "Stochastic_Proposed")]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(diag["hour"], diag["actual_renewable_scaled_mw"], color="#111827", label="실제")
    ax.plot(diag["hour"], diag["forecast_renewable_scaled_mw"], color="#2563eb", label="예측")
    ax.fill_between(
        diag["hour"].to_numpy(),
        (diag["forecast_renewable_scaled_mw"] - diag["sigma_u_scaled_mw"]).to_numpy(),
        (diag["forecast_renewable_scaled_mw"] + diag["sigma_u_scaled_mw"]).to_numpy(),
        color="#93c5fd",
        alpha=0.4,
        label="불확실성 밴드",
    )
    ax.set_xlabel("시간")
    ax.set_ylabel("MW")
    ax.set_title("Synthetic fallback uncertainty")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_uncertainty_risk.png", dpi=220)
    plt.close(fig)

    heat = (
        line_df[(line_df["subject_id"].astype(str) == seed) & (line_df["mode"] == "Stochastic_Proposed")]
        .pivot(index="line_id", columns="hour", values="utilization_pct")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=120)
    ax.set_xticks(range(0, 24, 2), range(0, 24, 2))
    ax.set_yticks(range(0, len(heat.index), 2), heat.index[::2])
    ax.set_xlabel("시간")
    ax.set_ylabel("송전선로 번호")
    ax.set_title("Synthetic fallback line loading")
    fig.colorbar(im, ax=ax).set_label("선로 이용률 (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_line_heatmap.png", dpi=220)
    plt.close(fig)


def print_console_summary(
    summary_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    rm_anova_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> None:
    table = summary_df.groupby(["analysis_type", "mode"]).agg(
        curtailment_mean=("total_curtailment_mwh", "mean"),
        curtailment_std=("total_curtailment_mwh", "std"),
        max_line_loading_mean=("max_line_loading", "mean"),
        ess_utilization_mean=("ess_utilization_pct", "mean"),
    )
    print("\n=== Simulation summary from generated data ===")
    print(table.round(3).to_string())
    print("\n=== ANOVA compatibility output ===")
    print(anova_df.round(8).to_string(index=False))
    print("\n=== Repeated-measures ANOVA ===")
    print(rm_anova_df.round(8).to_string(index=False))
    print("\n=== Paired tests ===")
    print(paired_df.round(8).to_string(index=False))
    print(f"\nData saved to: {OUTPUT_DATA_DIR}")
    print(f"Figures saved to: {FIG_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Jeju real-data VPP/ESS simulation")
    parser.add_argument("--synthetic", action="store_true", help="Run the old synthetic fallback simulation.")
    args = parser.parse_args()
    if args.synthetic:
        result = run_synthetic_simulation()
    else:
        result = run_real_data_simulation()
    print_console_summary(result[2], result[3], result[4], result[5])


if __name__ == "__main__":
    main()
