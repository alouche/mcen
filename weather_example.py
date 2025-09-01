from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

from mcen import (
    MCENSystem,
    ExponentialDecay,
    PowerLawDecay,
    BuildupPowerDecay,
    NonparametricKernelDecay,
)
import warnings

warnings.filterwarnings("ignore")


def fetch_weather_data(
    latitude: float = 41.8781, longitude: float = -87.6298, years_back: int = 2
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Climate API.
    """
    print("Fetching historical weather data from Open-Meteo Climate API...")
    url = "https://archive-api.open-meteo.com/v1/archive"

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * years_back)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ConnectionError(
            f"Failed to fetch  {response.status_code}\n{response.text}"
        )

    data = response.json()
    if "daily" not in data:
        raise ValueError("No daily data in response. Response: " + str(data))

    dates = pd.to_datetime(data["daily"]["time"]).date
    temp_mean = np.array(data["daily"]["temperature_2m_mean"])
    temp_max = np.array(data["daily"]["temperature_2m_max"])
    temp_min = np.array(data["daily"]["temperature_2m_min"])

    df = pd.DataFrame(
        {"Date": dates, "TempAvg": temp_mean, "TempMin": temp_min, "TempMax": temp_max}
    )
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Fetched {len(df)} days of historical weather data.")
    return df


def detect_cold_fronts(
    df: pd.DataFrame, window: int = 3, drop_threshold: float = 5.0
) -> List:
    """
    Detect cold front events as days with large temp drops over short period.
    Returns list of front dates with metadata.
    """
    print("Detecting cold front events...")
    df = df.copy()
    df["TempSmooth"] = (
        df["TempAvg"].rolling(window=window, center=True, min_periods=1).mean()
    )
    df["TempDrop"] = df["TempSmooth"].diff(periods=-window)

    front_dates = []
    i = 0
    while i < len(df) - window:
        if df.iloc[i]["TempDrop"] >= drop_threshold:
            window_end = min(i + window, len(df))
            peak_drop_idx = df.iloc[i:window_end]["TempDrop"].idxmax()
            front_date = df.iloc[peak_drop_idx]["Date"]
            front_dates.append(front_date)
            i = peak_drop_idx + window
        else:
            i += 1

    print(f"Detected {len(front_dates)} cold front events.")
    return front_dates


def build_weather_interventional_data(
    df: pd.DataFrame, front_dates: List, max_lag: int = 10, outcome_col: str = "TempAvg"
) -> pd.DataFrame:
    """
    Build interventional dataset with cold front and matched control days.
    """
    print("Building interventional dataset...")
    interventional_data = []
    front_set = set(front_dates)

    df["Season"] = pd.PeriodIndex(df["Date"], freq="Q")
    df["TempVol"] = df["TempAvg"].rolling(7, min_periods=3).std().fillna(0)

    event_indices = []
    for date in front_dates:
        matching = df[df["Date"] == date]
        if len(matching) > 0:
            event_indices.append(matching.index[0])

    control_candidates = df[~df["Date"].isin(front_set)].copy()
    eligible_controls = []

    for idx in event_indices:
        q = df.loc[idx, "Season"]
        vol = df.loc[idx, "TempVol"]
        candidates = control_candidates[
            (control_candidates["Season"] == q)
            & (control_candidates.index + max_lag < len(df))
        ]
        if len(candidates) == 0:
            continue
        candidates = candidates.copy()
        candidates["vol_diff"] = abs(candidates["TempVol"] - vol)
        best_match = candidates.loc[candidates["vol_diff"].idxmin()]
        eligible_controls.append(best_match["Date"])

    n_needed = min(len(event_indices), len(eligible_controls))
    control_dates = (
        np.random.choice(eligible_controls, size=n_needed, replace=False)
        if eligible_controls
        else []
    )

    for idx in event_indices:
        for lag in range(0, max_lag + 1):
            event_idx = idx + lag
            if event_idx < len(df):
                interventional_data.append(
                    {
                        "intervention": "cold_front",
                        "outcome": df.iloc[event_idx][outcome_col],
                        "time_lag": lag,
                        "event_date": df.iloc[idx]["Date"],
                    }
                )

    for ctrl_date in control_dates:
        idx = df[df["Date"] == ctrl_date].index[0]
        for lag in range(0, max_lag + 1):
            event_idx = idx + lag
            if event_idx < len(df):
                interventional_data.append(
                    {
                        "intervention": "control",
                        "outcome": df.iloc[event_idx][outcome_col],
                        "time_lag": lag,
                        "event_date": ctrl_date,
                    }
                )

    return pd.DataFrame(interventional_data)


def temporal_event_split(
    interventional_data: pd.DataFrame, cutoff_date: str = "2025-01-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data based on when the event (cold front or control) occurred.
    """
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    interventional_data["event_date"] = pd.to_datetime(
        interventional_data["event_date"]
    ).dt.date

    train_mask = interventional_data["event_date"] < cutoff
    test_mask = interventional_data["event_date"] >= cutoff

    train_data = interventional_data[train_mask].reset_index(drop=True)
    test_data = interventional_data[test_mask].reset_index(drop=True)

    print(f"\nTemporal Event Split (cutoff: {cutoff}):")
    print(f"  Training events: {train_data['event_date'].nunique()}")
    print(f"  Testing events: {test_data['event_date'].nunique()}")
    print(f"  Training data points: {len(train_data)}")
    print(f"  Testing data points: {len(test_data)}")

    return train_data, test_data


def evaluate_oos_accuracy(
    mcen, test_data, intervention="cold_front", baseline="control"
):
    """
    Evaluate model trained on train_data against test_data.
    """
    if intervention not in mcen.mchl_results:
        return {"error": "No model fitted"}

    result = mcen.mchl_results[intervention][0]
    pred_lags = result.time_lags
    pred_curve = result.effect_curve

    effects = mcen._compute_causal_effects_(
        test_data,
        intervention_col="intervention",
        outcome_col="outcome",
        time_col="time_lag",
        baseline_intervention=baseline,
    )
    if intervention not in effects:
        return {"error": "No test data for intervention"}

    true_lags = effects[intervention]["lags"]
    true_curve = effects[intervention]["effects"]

    from scipy.interpolate import interp1d

    try:
        f_pred = interp1d(
            pred_lags,
            pred_curve,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        pred_on_true = f_pred(true_lags)
    except Exception as e:
        return {"error": str(e)}

    rmse = np.sqrt(np.mean((true_curve - pred_on_true) ** 2))
    mae = np.mean(np.abs(true_curve - pred_on_true))

    try:
        peak_lag = int(true_lags[np.argmax(np.abs(true_curve))])
        treated = test_data[
            (test_data["intervention"] == intervention)
            & (test_data["time_lag"] == peak_lag)
        ]["outcome"]
        control = test_data[
            (test_data["intervention"] == baseline)
            & (test_data["time_lag"] == peak_lag)
        ]["outcome"]
        jsd = (
            mcen.information_metric.js_divergence(treated, control)
            if len(treated) > 5 and len(control) > 5
            else np.nan
        )
    except:
        jsd = np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "jsd": float(jsd) if not np.isnan(jsd) else None,
        "pred_lags": pred_lags.tolist(),
        "pred_curve": pred_curve.tolist(),
        "true_lags": true_lags.tolist(),
        "true_curve": true_curve.tolist(),
        "pred_on_true": pred_on_true.tolist(),
    }


def plot_oos_validation(metrics):
    if "error" in metrics:
        print("Plotting skipped due to error.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(
        metrics["true_lags"],
        metrics["true_curve"],
        "o-",
        label="Empirical (Test)",
        color="black",
        linewidth=2,
    )
    plt.plot(
        metrics["pred_lags"],
        metrics["pred_curve"],
        "--",
        label="MCEN Prediction (Train)",
        color="blue",
        alpha=0.8,
    )
    plt.plot(
        metrics["true_lags"],
        metrics["pred_on_true"],
        ":",
        label="Predicted (on test lags)",
        color="blue",
        alpha=0.6,
    )
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Days Since Cold Front")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.title("Out-of-Sample Validation: MCEN Prediction vs Test Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("MCEN WEATHER ANALYSIS: COLD FRONT IMPACT ON TEMPERATURE")
    print("=" * 60)

    df = fetch_weather_data(latitude=41.8781, longitude=-87.6298, years_back=2)

    front_dates = detect_cold_fronts(df, window=3, drop_threshold=5.0)
    if len(front_dates) < 3:
        print("Not enough cold fronts. Try a different location or threshold.")
        return

    interventional_data = build_weather_interventional_data(df, front_dates, max_lag=10)

    train_data, test_data = temporal_event_split(
        interventional_data, cutoff_date="2025-01-01"
    )
    if len(test_data) == 0:
        print("No test events. Adjust cutoff date.")
        return

    print("\nFitting MCEN on training data...")
    mcen = MCENSystem(variables=["cold_front", "temperature"], max_time_horizon=10)
    mcen._decay_models = [
        NonparametricKernelDecay(bandwidth_grid=None, allow_k0=True),
        BuildupPowerDecay(),
        PowerLawDecay(),
        ExponentialDecay(),
    ]

    _ = mcen.learn__mchl_profiles(
        interventional_data=train_data,
        intervention_col="intervention",
        outcome_col="outcome",
        time_col="time_lag",
        baseline_intervention="control",
    )

    if "cold_front" not in mcen.mchl_results:
        print("ERROR: No model fitted on training data.")
        return

    result = mcen.mchl_results["cold_front"][0]
    print(f"Trained model: {result.decay_type} (P={result.posterior_prob:.3f})")

    print("\nEvaluating on test data...")
    oos_metrics = evaluate_oos_accuracy(mcen, test_data)

    if "error" not in oos_metrics:
        print(f"- RMSE: {oos_metrics['rmse']:.4f}")
        print(f"- MAE:  {oos_metrics['mae']:.4f}")
        jsd_val = oos_metrics["jsd"] if oos_metrics["jsd"] is not None else "N/A"
        print(f"- JSD (causal divergence): {jsd_val}")
        plot_oos_validation(oos_metrics)
    else:
        print(f"Evaluation failed: {oos_metrics['error']}")

    print("\nValidating theoretical properties...")
    validation = mcen.validate_theoretical_properties()
    score = validation["cold_front"]["profile_0"]["identifiability_score"]
    conv_rate = validation["cold_front"]["profile_0"]["convergence_rate"]
    tail = validation["cold_front"]["profile_0"]["tail_bound"]
    print(f"Identifiability Score: {score:.2f}")
    print(f"Convergence Rate: {conv_rate:.3f}")
    print(f"Tail Bound: {tail:.4f}")

    print("Simulating response...")
    mcen.setup_convolutional_dynamics(outcome_var="temperature", min_r2_for_param=0.1)
    scenario = {
        "intervention_var": "cold_front",
        "intervention_value": 1.0,
        "initial_state": {"cold_front": 1.0, "temperature": df["TempAvg"].mean()},
    }
    sim = mcen.simulate_with_convolution(scenario, time_horizon=15)
    plt.figure(figsize=(10, 5))
    plt.plot(sim["temperature"], "o-", label="Simulated Temp Response")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Days")
    plt.ylabel("Temperature Response (°C)")
    plt.title("Simulated Response to Cold Front")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    mcen.export_theoretical_analysis("cold_front_mcen_analysis.pkl")
    print("\nAnalysis complete and exported.")

if __name__ == "__main__":
    main()
