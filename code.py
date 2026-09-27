import re
import time
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = "/app/data/dataset_2026-09-27T11_22_24.528152343Z_DEFAULT_INTEGRATION_IMF.STA_ER_4.0.1.csv"
OUT_PATH = "/app/output/forecasts_all_currencies.csv"

# Minimum number of usable observations required for forecasting
MIN_OBS = 24

# Number of lagged observations used as model features
# Features = t-1, t-2, ..., t-N_LAGS
N_LAGS = 6

# Number of periods to forecast
HORIZON = 6

# Set to an integer for a quick test run, or None for all series
MAX_SERIES = None

# Example: {"Monthly"}
# None = use Monthly, Quarterly and Annual according to availability
PREFERRED_FREQUENCIES = None

# Number of multiprocessing workers
N_WORKERS = max(1, cpu_count() - 1)


# ============================================================================
# DATASET COLUMNS
# ============================================================================

META_COLS = [
    "DATASET",
    "SERIES_CODE",
    "OBS_MEASURE",
    "COUNTRY",
    "INDICATOR",
    "TYPE_OF_TRANSFORMATION",
    "FREQUENCY",
    "SCALE",
    "DECIMALS_DISPLAYED",
    "EXRATE",
    "STATISTICAL_MEASURES",
    "TRANSFORMATION",
    "OVERLAP",
    "IFS_FLAG",
    "DOI",
    "FULL_DESCRIPTION",
    "PUBLISHER",
    "DEPARTMENT",
    "CONTACT_POINT",
    "TOPIC",
    "TOPIC_DATASET",
    "KEYWORDS",
    "KEYWORDS_DATASET",
    "LANGUAGE",
    "PUBLICATION_DATE",
    "UPDATE_DATE",
    "METHODOLOGY",
    "METHODOLOGY_NOTES",
    "ACCESS_SHARING_LEVEL",
    "ACCESS_SHARING_NOTES",
    "SECURITY_CLASSIFICATION",
    "SHORT_SOURCE_CITATION",
    "FULL_SOURCE_CITATION",
    "LICENSE",
    "SUGGESTED_CITATION",
    "KEY_INDICATOR",
    "SERIES_NAME",
]

ID_COLS = [
    "SERIES_CODE",
    "COUNTRY",
    "INDICATOR",
    "TYPE_OF_TRANSFORMATION",
    "FREQUENCY",
]


# ============================================================================
# FREQUENCY SETTINGS
# ============================================================================

FREQ_ALIAS = {
    "Annual": "YS",
    "Quarterly": "QS",
    "Monthly": "MS",
}


# ============================================================================
# PERIOD PARSING
# ============================================================================

def parse_period(period):
    """
    Convert IMF period strings into pandas timestamps.

    Supported formats:
        YYYY
        YYYY-Q1 ... YYYY-Q4
        YYYY-M01 ... YYYY-M12
    """

    period = str(period)

    # Annual
    if re.fullmatch(r"\d{4}", period):
        return pd.Timestamp(int(period), 1, 1)

    # Quarterly
    match = re.fullmatch(r"(\d{4})-Q([1-4])", period)

    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))

        month = (quarter - 1) * 3 + 1

        return pd.Timestamp(year, month, 1)

    # Monthly
    match = re.fullmatch(r"(\d{4})-M(\d{2})", period)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))

        return pd.Timestamp(year, month, 1)

    return pd.NaT


# ============================================================================
# LOAD DATA
# ============================================================================

def load_long(csv_path=CSV_PATH):
    """
    Load the IMF CSV and convert it from wide to long format.

    Returns
    -------
    pandas.DataFrame
        Columns include:
            SERIES_CODE
            COUNTRY
            INDICATOR
            TYPE_OF_TRANSFORMATION
            FREQUENCY
            PERIOD_RAW
            VALUE
            DATE
    """

    print("Reading CSV...")

    df = pd.read_csv(
        csv_path,
        low_memory=False
    )

    # Everything not identified as metadata is treated as a period column
    date_cols = [
        column
        for column in df.columns
        if column not in META_COLS
    ]

    print("Melting to long format...")

    long_df = df[
        ID_COLS + date_cols
    ].melt(
        id_vars=ID_COLS,
        var_name="PERIOD_RAW",
        value_name="VALUE"
    )

    # Remove missing values
    long_df = long_df.dropna(
        subset=["VALUE"]
    )

    # Remove empty strings
    long_df = long_df[
        long_df["VALUE"] != ""
    ]

    # Convert observations to numeric
    long_df["VALUE"] = pd.to_numeric(
        long_df["VALUE"],
        errors="coerce"
    )

    long_df = long_df.dropna(
        subset=["VALUE"]
    )

    print("Parsing periods...")

    long_df["DATE"] = (
        long_df["PERIOD_RAW"]
        .apply(parse_period)
    )

    long_df = long_df.dropna(
        subset=["DATE"]
    )

    # Sort chronologically within each series
    long_df = long_df.sort_values(
        ["SERIES_CODE", "DATE"]
    )

    return long_df


# ============================================================================
# SELECT CANONICAL SERIES
# ============================================================================

def select_canonical_series(
    long_df,
    meta_df,
    base_currency="US Dollar",
    direction_prefix="Domestic currency per",
    preferred_transformation="End-of-period (EoP)"
):
    """
    Select one exchange-rate series per country.

    Selection rules:

    1. Only use:
           Domestic currency per US Dollar

    2. Prefer the highest available frequency:
           Monthly > Quarterly > Annual

    3. If multiple series have the same frequency,
       prefer End-of-period (EoP).

    This produces approximately one usable series per country.
    """

    indicator = (
        f"{direction_prefix} {base_currency}"
    )

    candidates = meta_df[
        meta_df["INDICATOR"] == indicator
    ].copy()

    # Lower number = higher priority
    frequency_rank = {
        "Monthly": 0,
        "Quarterly": 1,
        "Annual": 2,
    }

    candidates["FREQ_RANK"] = (
        candidates["FREQUENCY"]
        .map(frequency_rank)
    )

    # Prefer EoP when frequency is tied
    candidates["TRANSFORM_RANK"] = (
        candidates["TYPE_OF_TRANSFORMATION"]
        != preferred_transformation
    ).astype(int)

    best = (
        candidates
        .sort_values(
            [
                "FREQ_RANK",
                "TRANSFORM_RANK"
            ]
        )
        .drop_duplicates(
            subset="COUNTRY",
            keep="first"
        )
        .drop(
            columns=[
                "FREQ_RANK",
                "TRANSFORM_RANK"
            ]
        )
    )

    return best.reset_index(
        drop=True
    )


# ============================================================================
# LAG FEATURE CREATION
# ============================================================================

def make_lag_features(values, n_lags):
    """
    Convert a time series into supervised-learning features.

    Example with N_LAGS = 3:

        X = [t-1, t-2, t-3]
        y = t

    The most recent lag is placed first.
    """

    X = []
    y = []

    for i in range(
        n_lags,
        len(values)
    ):
        X.append(
            values[
                i - n_lags:i
            ][::-1]
        )

        y.append(
            values[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================================
# GBR MODEL
# ============================================================================

def build_model():
    """
    Create the Gradient Boosting Regression model.

    These parameters are intentionally kept relatively conservative
    because each currency series may contain only a modest number
    of observations.
    """

    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=0
    )


# ============================================================================
# FORECAST ONE SERIES
# ============================================================================

def fit_forecast_one(
    ts,
    freq_alias
):
    """
    Fit GBR to one currency series and forecast HORIZON periods.

    Returns
    -------
    forecast_series
    lower_confidence_interval
    upper_confidence_interval
    model_label
    """

    # Remove duplicate dates
    ts = ts[
        ~ts.index.duplicated(
            keep="last"
        )
    ]

    # Put the series on its expected frequency
    ts = ts.asfreq(
        freq_alias
    )

    # Fill internal missing values only
    ts = ts.interpolate(
        limit_area="inside"
    )

    ts_clean = ts.dropna()

    # ---------------------------------------------------------------------
    # Check minimum history
    # ---------------------------------------------------------------------

    if len(ts_clean) < MIN_OBS:

        return (
            None,
            None,
            None,
            "skipped_too_short"
        )

    values = (
        ts_clean
        .values
        .astype(float)
    )

    # ---------------------------------------------------------------------
    # Constant series
    # ---------------------------------------------------------------------

    if np.std(values) < 1e-12:

        last_value = values[-1]

        index = pd.date_range(
            ts_clean.index[-1],
            periods=HORIZON + 1,
            freq=freq_alias
        )[1:]

        forecast = pd.Series(
            [last_value] * HORIZON,
            index=index
        )

        return (
            forecast,
            forecast.copy(),
            forecast.copy(),
            "constant"
        )

    # ---------------------------------------------------------------------
    # Determine number of lags
    # ---------------------------------------------------------------------

    n_lags = min(
        N_LAGS,
        len(values) - 2
    )

    if n_lags < 1:

        return (
            None,
            None,
            None,
            "skipped_too_short"
        )

    # ---------------------------------------------------------------------
    # Train GBR
    # ---------------------------------------------------------------------

    try:

        X, y = make_lag_features(
            values,
            n_lags
        )

        model = build_model()

        model.fit(
            X,
            y
        )

        # -------------------------------------------------------------
        # Estimate residual uncertainty
        # -------------------------------------------------------------

        fitted = model.predict(X)

        residuals = y - fitted

        if len(residuals) > 1:
            residual_std = np.std(
                residuals
            )
        else:
            residual_std = (
                abs(values[-1]) * 0.05
            )

        # -------------------------------------------------------------
        # Recursive forecasting
        # -------------------------------------------------------------

        history = list(
            values[-n_lags:]
        )

        predictions = []

        for _ in range(HORIZON):

            x_input = np.array(
                history[-n_lags:][::-1]
            ).reshape(
                1,
                -1
            )

            next_value = model.predict(
                x_input
            )[0]

            predictions.append(
                next_value
            )

            # Feed prediction back into history
            history.append(
                next_value
            )

        # -------------------------------------------------------------
        # Forecast dates
        # -------------------------------------------------------------

        index = pd.date_range(
            ts_clean.index[-1],
            periods=HORIZON + 1,
            freq=freq_alias
        )[1:]

        steps = np.arange(
            1,
            HORIZON + 1
        )

        forecast = pd.Series(
            predictions,
            index=index
        )

        # Rough uncertainty interval
        lower = (
            forecast
            - 1.96
            * residual_std
            * np.sqrt(steps)
        )

        upper = (
            forecast
            + 1.96
            * residual_std
            * np.sqrt(steps)
        )

        return (
            forecast,
            lower,
            upper,
            f"gbr_lag{n_lags}"
        )

    # ---------------------------------------------------------------------
    # GBR failed
    # ---------------------------------------------------------------------

    except Exception:

        pass

    # =========================================================================
    # FALLBACK: NAIVE DRIFT
    # =========================================================================

    try:

        differences = np.diff(
            values
        )

        if len(differences):

            drift = (
                differences[
                    -min(
                        len(differences),
                        12
                    ):
                ]
                .mean()
            )

        else:

            drift = 0.0

        last_value = values[-1]

        index = pd.date_range(
            ts_clean.index[-1],
            periods=HORIZON + 1,
            freq=freq_alias
        )[1:]

        steps = np.arange(
            1,
            HORIZON + 1
        )

        forecast_values = (
            last_value
            + drift * steps
        )

        if len(differences) > 1:

            residual_std = np.std(
                differences
            )

        else:

            residual_std = (
                abs(last_value) * 0.05
            )

        forecast = pd.Series(
            forecast_values,
            index=index
        )

        lower = (
            forecast
            - 1.96
            * residual_std
            * np.sqrt(steps)
        )

        upper = (
            forecast
            + 1.96
            * residual_std
            * np.sqrt(steps)
        )

        return (
            forecast,
            lower,
            upper,
            "naive_drift"
        )

    except Exception:

        return (
            None,
            None,
            None,
            "failed"
        )


# ============================================================================
# PROCESS ONE SERIES
# ============================================================================

def _process_one(args):
    """
    Worker function used by multiprocessing.
    """

    row_dict, ts = args

    freq_alias = FREQ_ALIAS.get(
        row_dict["FREQUENCY"],
        "MS"
    )

    (
        forecast,
        lower,
        upper,
        model_label
    ) = fit_forecast_one(
        ts,
        freq_alias
    )

    output_rows = []

    # ---------------------------------------------------------------------
    # Failed / skipped series
    # ---------------------------------------------------------------------

    if forecast is None:

        output_rows.append({

            **row_dict,

            "N_OBS": len(ts),

            "MODEL": model_label,

            "LAST_DATE": (
                ts.index.max()
                if len(ts)
                else pd.NaT
            ),

            "LAST_VALUE": (
                ts.iloc[-1]
                if len(ts)
                else np.nan
            ),
        })

        return output_rows

    # ---------------------------------------------------------------------
    # Successful forecast
    # ---------------------------------------------------------------------

    n_obs = len(
        ts.dropna()
    )

    last_date = ts.index.max()

    last_value = ts.iloc[-1]

    for step, (
        forecast_date,
        forecast_value
    ) in enumerate(
        forecast.items(),
        start=1
    ):

        output_rows.append({

            **row_dict,

            "N_OBS": n_obs,

            "MODEL": model_label,

            "LAST_DATE": last_date,

            "LAST_VALUE": last_value,

            "FORECAST_STEP": step,

            "FORECAST_DATE": forecast_date,

            "FORECAST_VALUE": forecast_value,

            "LOWER_CI": (
                lower.iloc[step - 1]
                if lower is not None
                else np.nan
            ),

            "UPPER_CI": (
                upper.iloc[step - 1]
                if upper is not None
                else np.nan
            ),
        })

    return output_rows


# ============================================================================
# MAIN BATCH FORECAST
# ============================================================================

def run_batch(parallel=True):
    """
    Run GBR forecasting for all selected currency series.

    Returns
    -------
    pandas.DataFrame
        Forecast results.
    """

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------

    long_df = load_long()

    # ---------------------------------------------------------------------
    # Build series metadata
    # ---------------------------------------------------------------------

    series_meta = (
        long_df[ID_COLS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    series_meta = select_canonical_series(
        long_df,
        series_meta
    )

    # Optional frequency filter
    if PREFERRED_FREQUENCIES:

        series_meta = series_meta[
            series_meta["FREQUENCY"].isin(
                PREFERRED_FREQUENCIES
            )
        ]

    # Optional series limit
    if MAX_SERIES:

        series_meta = (
            series_meta
            .head(MAX_SERIES)
        )

    print(
        f"Forecasting {len(series_meta)} series "
        f"using GradientBoostingRegressor "
        f"(min_obs={MIN_OBS}, "
        f"n_lags={N_LAGS}, "
        f"horizon={HORIZON}, "
        f"workers={N_WORKERS if parallel else 1})..."
    )

    # ---------------------------------------------------------------------
    # Group time series
    # ---------------------------------------------------------------------

    grouped = {
        code: group
        .set_index("DATE")["VALUE"]

        for code, group
        in long_df.groupby(
            "SERIES_CODE"
        )
    }

    # ---------------------------------------------------------------------
    # Build worker tasks
    # ---------------------------------------------------------------------

    tasks = []

    for _, row in series_meta.iterrows():

        series_code = row[
            "SERIES_CODE"
        ]

        ts = grouped.get(
            series_code
        )

        if ts is not None and not ts.empty:

            tasks.append(
                (
                    row.to_dict(),
                    ts
                )
            )

    # ---------------------------------------------------------------------
    # Run forecasts
    # ---------------------------------------------------------------------

    start_time = time.time()

    rows = []

    if parallel and N_WORKERS > 1:

        with Pool(N_WORKERS) as pool:

            for i, result_rows in enumerate(
                pool.imap(
                    _process_one,
                    tasks,
                    chunksize=20
                ),
                start=1
            ):

                rows.extend(
                    result_rows
                )

                if i % 100 == 0:

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    print(
                        f"  {i}/{len(tasks)} "
                        f"series done "
                        f"({elapsed:.0f}s elapsed)"
                    )

    else:

        for i, task in enumerate(
            tasks,
            start=1
        ):

            rows.extend(
                _process_one(task)
            )

            if i % 100 == 0:

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"  {i}/{len(tasks)} "
                    f"series done "
                    f"({elapsed:.0f}s elapsed)"
                )

    # ---------------------------------------------------------------------
    # Create output DataFrame
    # ---------------------------------------------------------------------

    result = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------------------
    # Save forecasts
    # ---------------------------------------------------------------------

    result.to_csv(
        OUT_PATH,
        index=False
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nDone in {elapsed:.0f}s."
    )

    print(
        f"Wrote {len(result):,} rows "
        f"to {OUT_PATH}"
    )

    return result


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    run_batch()
