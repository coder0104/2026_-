from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TIME_HORIZON = 24
DT = 1.0
N_NODES = 14
LINE_TRIP_LIMIT = 80.0
CAP_RENEWABLE = 200.0
ESS_TOTAL_CAP = 400.0
ESS_EFF = 0.95
N_SEEDS = 10

MODES = [
    "Heuristic_Rule",
    "Deterministic_Opt",
    "Rolling_Greedy",
    "Stochastic_Proposed",
]

OUT_DIR = Path("outputs")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"


@dataclass(frozen=True)
class Scenario:
    forecast_renewable: np.ndarray
    actual_renewable: np.ndarray
    sigma_u: np.ndarray
    load_by_node: np.ndarray
    renewable_share_by_node: np.ndarray


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


def make_scenario(seed: int) -> Scenario:
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
    sigma_pct = 0.10 + 0.25 * cloud_volatility
    sigma_u = forecast_renewable * sigma_pct
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
        load_by_node=load_by_node,
        renewable_share_by_node=renewable_share_by_node,
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


def forecast_line_stress(scenario: Scenario, hour: int, soc: np.ndarray) -> float:
    del soc
    injection = make_injection(
        scenario.forecast_renewable[hour],
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


def price_curve() -> np.ndarray:
    hours = np.arange(TIME_HORIZON)
    return 72 + 18 * np.exp(-((hours - 9) / 3.4) ** 2) + 43 * np.exp(-((hours - 19) / 2.8) ** 2)


def dispatch_action(mode: str, scenario: Scenario, hour: int, soc: np.ndarray, det_plan: np.ndarray) -> tuple[np.ndarray, float]:
    total_load = scenario.load_by_node[hour].sum()
    forecast = scenario.forecast_renewable[hour]
    actual = scenario.actual_renewable[hour]
    sigma = scenario.sigma_u[hour]
    price = price_curve()[hour]

    action = np.zeros(len(ESS_NODES))
    predicted_stress = forecast_line_stress(scenario, hour, soc)
    risk_score = float(np.exp((predicted_stress + sigma) / LINE_TRIP_LIMIT))

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
        action = stochastic_objective_dispatch(scenario, hour, soc)

    return apply_ess_limits(action, soc), risk_score


def stochastic_objective_dispatch(scenario: Scenario, hour: int, soc: np.ndarray) -> np.ndarray:
    local_trial = make_injection(
        scenario.actual_renewable[hour],
        scenario.renewable_share_by_node[hour],
        scenario.load_by_node[hour],
        np.zeros(len(ESS_NODES)),
    )
    local_excess = np.clip(local_trial[RENEWABLE_NODES] - LOCAL_RENEWABLE_EXPORT_LIMIT * 0.88, 0, None)
    future_need = forecast_absorption_need(scenario, hour)
    current_room = ESS_NODE_CAP - soc
    reserve_shortage_now = np.clip(future_need - current_room, 0, None)
    price = price_curve()[hour]

    absorb_action = -np.minimum(ESS_NODE_POWER, local_excess)
    reserve_action = np.minimum(ESS_NODE_POWER, reserve_shortage_now * ESS_EFF)
    mixed_action = np.where(local_excess > 0, absorb_action, reserve_action)
    economic_action = np.minimum(ESS_NODE_POWER * 0.45, soc * ESS_EFF * 0.18) if price > 100 else np.zeros(len(ESS_NODES))
    candidates = [
        np.zeros(len(ESS_NODES)),
        absorb_action,
        reserve_action,
        mixed_action,
        economic_action,
    ]

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


def forecast_absorption_need(scenario: Scenario, hour: int) -> np.ndarray:
    need = np.zeros(len(ESS_NODES))
    for h in range(hour + 1, min(TIME_HORIZON, hour + 1 + STOCHASTIC_LOOKAHEAD)):
        risk_renewable = scenario.forecast_renewable[h] + STOCHASTIC_RISK_Z * scenario.sigma_u[h]
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
        trial_injection = make_injection(
            trial_renewable,
            renewable_share,
            load_nodes,
            ess_action,
        )
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


def curtail_to_safe_limit(scenario: Scenario, hour: int, ess_action: np.ndarray) -> tuple[float, np.ndarray]:
    return curtailment_for_renewable(
        scenario.actual_renewable[hour],
        scenario.renewable_share_by_node[hour],
        scenario.load_by_node[hour],
        ess_action,
    )


def simulate_mode(seed: int, mode: str, scenario: Scenario) -> tuple[list[dict], list[dict], dict]:
    soc = np.full(len(ESS_NODES), ESS_NODE_CAP * 0.45)
    det_plan = deterministic_plan(scenario)
    rows = []
    line_rows = []
    total_curtailment = 0.0
    total_revenue = 0.0
    throughput = 0.0
    max_overload_before = 0.0

    for hour in range(TIME_HORIZON):
        action, risk_score = dispatch_action(mode, scenario, hour, soc, det_plan)
        before_injection = make_injection(
            scenario.actual_renewable[hour],
            scenario.renewable_share_by_node[hour],
            scenario.load_by_node[hour],
            action,
        )
        flows_before = dc_power_flow(before_injection)
        overload_pct = max(0.0, np.max(np.abs(flows_before)) / LINE_TRIP_LIMIT - 1.0) * 100
        max_overload_before = max(max_overload_before, overload_pct)

        curtailment, flows_after = curtail_to_safe_limit(scenario, hour, action)
        renewable_used = scenario.actual_renewable[hour] - curtailment
        price = price_curve()[hour]
        discharge = np.clip(action, 0, None).sum()
        charge = np.clip(-action, 0, None).sum()
        revenue = (renewable_used * price + discharge * price * 0.35 - charge * price * 0.04) / 100.0

        total_curtailment += curtailment * DT
        total_revenue += revenue
        throughput += (charge + discharge) * DT
        soc = update_soc(soc, action)

        rows.append(
            {
                "seed": seed,
                "mode": mode,
                "hour": hour,
                "forecast_renewable_mw": scenario.forecast_renewable[hour],
                "actual_renewable_mw": scenario.actual_renewable[hour],
                "sigma_u_mw": scenario.sigma_u[hour],
                "total_load_mw": scenario.load_by_node[hour].sum(),
                "ess_charge_mw": charge,
                "ess_discharge_mw": discharge,
                "soc_total_mwh": soc.sum(),
                "soc_percent": soc.sum() / ESS_TOTAL_CAP * 100,
                "curtailment_mwh": curtailment * DT,
                "max_line_flow_before_mw": np.max(np.abs(flows_before)),
                "max_line_flow_after_mw": np.max(np.abs(flows_after)),
                "overload_before_pct": overload_pct,
                "risk_score": risk_score,
                "revenue_10k_krw": revenue,
            }
        )
        for line_id, flow in enumerate(flows_after, start=1):
            line_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "hour": hour,
                    "line_id": line_id,
                    "flow_mw": flow,
                    "abs_flow_mw": abs(flow),
                    "utilization_pct": abs(flow) / LINE_TRIP_LIMIT * 100,
                }
            )

    summary = {
        "seed": seed,
        "mode": mode,
        "daily_curtailment_mwh": total_curtailment,
        "max_overload_before_pct": max_overload_before,
        "ess_utilization_pct": throughput / ESS_TOTAL_CAP * 100,
        "vpp_revenue_10k_krw": total_revenue,
    }
    return rows, line_rows, summary


def repeated_measures_anova(summary_df: pd.DataFrame) -> pd.DataFrame:
    wide = summary_df.pivot(index="seed", columns="mode", values="daily_curtailment_mwh").reindex(columns=MODES)
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
    ms_error = ss_error / df_error
    f_stat = ms_conditions / ms_error
    p_value = stats.f.sf(f_stat, df_conditions, df_error)
    partial_eta_squared = ss_conditions / (ss_conditions + ss_error)

    return pd.DataFrame(
        [
            {
                "metric": "daily_curtailment_mwh",
                "factor": "mode",
                "ss_factor": ss_conditions,
                "df_factor": df_conditions,
                "ms_factor": ms_conditions,
                "ss_subject_seed": ss_subjects,
                "df_subject_seed": df_subjects,
                "ss_error": ss_error,
                "df_error": df_error,
                "ms_error": ms_error,
                "f_statistic": f_stat,
                "p_value": p_value,
                "partial_eta_squared": partial_eta_squared,
                "note": "Repeated-measures ANOVA with weather seed as the repeated subject.",
            }
        ]
    )


def run_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    time_rows = []
    line_rows = []
    summary_rows = []
    for seed in range(1, N_SEEDS + 1):
        scenario = make_scenario(seed)
        for mode in MODES:
            t_rows, l_rows, summary = simulate_mode(seed, mode, scenario)
            time_rows.extend(t_rows)
            line_rows.extend(l_rows)
            summary_rows.append(summary)

    time_df = pd.DataFrame(time_rows)
    line_df = pd.DataFrame(line_rows)
    summary_df = pd.DataFrame(summary_rows)

    groups = [summary_df.loc[summary_df["mode"] == mode, "daily_curtailment_mwh"].values for mode in MODES]
    f_stat, p_value = stats.f_oneway(*groups)
    anova_df = pd.DataFrame(
        [
            {
                "metric": "daily_curtailment_mwh",
                "f_statistic": f_stat,
                "p_value": p_value,
                "note": "Independent one-way ANOVA. This ignores shared weather seeds.",
            }
        ]
    )
    wide = summary_df.pivot(index="seed", columns="mode", values="daily_curtailment_mwh")
    paired_rows = []
    for baseline in ["Heuristic_Rule", "Deterministic_Opt", "Rolling_Greedy"]:
        diff = wide[baseline] - wide["Stochastic_Proposed"]
        t_result = stats.ttest_rel(wide[baseline], wide["Stochastic_Proposed"])
        w_result = stats.wilcoxon(wide[baseline], wide["Stochastic_Proposed"])
        paired_rows.append(
            {
                "metric": "daily_curtailment_mwh",
                "baseline_mode": baseline,
                "proposed_mode": "Stochastic_Proposed",
                "mean_baseline_minus_proposed_mwh": diff.mean(),
                "paired_t_statistic": t_result.statistic,
                "paired_t_p_value": t_result.pvalue,
                "wilcoxon_statistic": w_result.statistic,
                "wilcoxon_p_value": w_result.pvalue,
                "n_paired_seeds": len(wide),
                "note": "Paired comparison by identical weather seed.",
            }
        )
    paired_df = pd.DataFrame(paired_rows)
    rm_anova_df = repeated_measures_anova(summary_df)

    time_df.to_csv(DATA_DIR / "time_series.csv", index=False)
    line_df.to_csv(DATA_DIR / "line_flows.csv", index=False)
    summary_df.to_csv(DATA_DIR / "simulation_summary.csv", index=False)
    anova_df.to_csv(DATA_DIR / "anova_result.csv", index=False)
    rm_anova_df.to_csv(DATA_DIR / "repeated_measures_anova.csv", index=False)
    paired_df.to_csv(DATA_DIR / "paired_tests.csv", index=False)
    return time_df, line_df, summary_df, anova_df, rm_anova_df, paired_df


def plot_figures(time_df: pd.DataFrame, line_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    configure_korean_font()
    colors = {
        "Heuristic_Rule": "#6b7280",
        "Deterministic_Opt": "#2563eb",
        "Rolling_Greedy": "#f97316",
        "Stochastic_Proposed": "#16a34a",
    }
    labels = {
        "Heuristic_Rule": "규칙 기반",
        "Deterministic_Opt": "확정 최적화",
        "Rolling_Greedy": "실시간 탐욕",
        "Stochastic_Proposed": "확률 리스크 인지",
    }

    stats_df = (
        summary_df.groupby("mode")["daily_curtailment_mwh"]
        .agg(["mean", "std"])
        .reindex(MODES)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(MODES))
    ax.bar(x, stats_df["mean"], yerr=stats_df["std"], capsize=6, color=[colors[m] for m in MODES])
    ax.set_xticks(x, [labels[m] for m in MODES])
    ax.set_ylabel("일평균 출력 제한량 (MWh)")
    ax.set_title("알고리즘 유형별 일평균 출력 제한 매몰량 비교")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_curtailment_bar.png", dpi=220)
    plt.close(fig)

    seed = 1
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for mode in MODES:
        subset = time_df[(time_df["seed"] == seed) & (time_df["mode"] == mode)]
        ax.plot(subset["hour"], subset["soc_percent"], marker="o", linewidth=2, color=colors[mode], label=labels[mode])
    ax.set_xlabel("시간 (시)")
    ax.set_ylabel("ESS 충전율 SOC (%)")
    ax.set_title("24시간 타임라인 기반 ESS 충전율 변화 궤적")
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_soc_timeseries.png", dpi=220)
    plt.close(fig)

    diag = time_df[(time_df["seed"] == seed) & (time_df["mode"] == "Stochastic_Proposed")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(diag["hour"], diag["forecast_renewable_mw"], color="#2563eb", label="AI 평균 예측")
    axes[0].fill_between(
        diag["hour"].to_numpy(),
        (diag["forecast_renewable_mw"] - diag["sigma_u_mw"]).to_numpy(),
        (diag["forecast_renewable_mw"] + diag["sigma_u_mw"]).to_numpy(),
        color="#93c5fd",
        alpha=0.42,
        label="불확실성 밴드 (+/- sigma_u)",
    )
    axes[0].plot(diag["hour"], diag["actual_renewable_mw"], color="#111827", linewidth=1.7, label="실제 발전량")
    axes[0].set_ylabel("발전량 (MW)")
    axes[0].set_title("AI 불확실성 오차 밴드폭과 실제 발전량")
    axes[0].grid(alpha=0.22)
    axes[0].legend(ncols=3, fontsize=9)

    axes[1].plot(diag["hour"], diag["risk_score"], color="#dc2626", marker="o", linewidth=2)
    axes[1].set_xlabel("시간 (시)")
    axes[1].set_ylabel("동적 리스크 점수")
    axes[1].set_title("VPP 동적 리스크 점수 변화")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_uncertainty_risk.png", dpi=220)
    plt.close(fig)

    heat = (
        line_df[(line_df["seed"] == seed) & (line_df["mode"] == "Stochastic_Proposed")]
        .pivot(index="line_id", columns="hour", values="utilization_pct")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=120)
    ax.set_xticks(range(0, 24, 2), range(0, 24, 2))
    ax.set_yticks(range(0, len(heat.index), 2), heat.index[::2])
    ax.set_xlabel("시간 (시)")
    ax.set_ylabel("송전선로 번호")
    ax.set_title("20개 송전선로의 24시간 시공간 부하 혼잡도 분포")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("선로 이용률 (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_line_heatmap.png", dpi=220)
    plt.close(fig)


def print_console_summary(
    summary_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    rm_anova_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> None:
    table = summary_df.groupby("mode").agg(
        daily_curtailment_mean=("daily_curtailment_mwh", "mean"),
        daily_curtailment_std=("daily_curtailment_mwh", "std"),
        max_overload_mean=("max_overload_before_pct", "mean"),
        ess_utilization_mean=("ess_utilization_pct", "mean"),
        revenue_mean=("vpp_revenue_10k_krw", "mean"),
    )
    print("\n=== Simulation summary from generated data ===")
    print(table.round(3).to_string())
    print("\n=== One-way ANOVA: daily_curtailment_mwh ===")
    print(anova_df.round(8).to_string(index=False))
    print("\n=== Repeated-measures ANOVA: daily_curtailment_mwh ===")
    print(rm_anova_df.round(8).to_string(index=False))
    print("\n=== Paired tests by identical weather seed ===")
    print(paired_df.round(8).to_string(index=False))
    print(f"\nData saved to: {DATA_DIR}")
    print(f"Figures saved to: {FIG_DIR}")


def main() -> None:
    time_df, line_df, summary_df, anova_df, rm_anova_df, paired_df = run_all()
    plot_figures(time_df, line_df, summary_df)
    print_console_summary(summary_df, anova_df, rm_anova_df, paired_df)


if __name__ == "__main__":
    main()
