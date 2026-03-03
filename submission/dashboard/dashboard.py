from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)


@st.cache_data(show_spinner=False)
def _ensure_data_files() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    day_path = data_dir / "day.csv"
    hour_path = data_dir / "hour.csv"
    if day_path.exists() and hour_path.exists():
        return day_path, hour_path

    with urlopen(DATA_URL) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for target_name in ("day.csv", "hour.csv"):
            member = next((n for n in zf.namelist() if n.endswith(f"/{target_name}") or n == target_name), None)
            if member is None:
                continue
            with zf.open(member) as src, open(data_dir / target_name, "wb") as dst:
                dst.write(src.read())

    if not day_path.exists() or not hour_path.exists():
        raise FileNotFoundError(
            "Gagal menyiapkan dataset. Pastikan file day.csv dan hour.csv ada di folder ./data "
            "atau koneksi internet tersedia untuk mengunduh dari UCI."
        )

    return day_path, hour_path


@st.cache_data(show_spinner=False)
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    day_path, hour_path = _ensure_data_files()

    day_df = pd.read_csv(day_path)
    hour_df = pd.read_csv(hour_path)

    day_df["dteday"] = pd.to_datetime(day_df["dteday"])
    hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])

    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_map = {
        1: "Clear/Partly Cloudy",
        2: "Mist/Cloudy",
        3: "Light Snow/Rain",
        4: "Heavy Rain/Ice",
    }

    for df in (day_df, hour_df):
        df["season_label"] = df["season"].map(season_map)
        df["weather_label"] = df["weathersit"].map(weather_map)
        df["year"] = df["yr"] + 2011
        df["temp_c"] = df["temp"] * 41

    day_df["year_month"] = day_df["dteday"].dt.to_period("M").astype(str)

    weekday_map = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
    }
    hour_df["weekday_label"] = hour_df["weekday"].map(weekday_map)

    return day_df, hour_df


def _monthly_avg_pivot(day_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    monthly_avg_df = (
        day_df.groupby(["year_month", "year"], as_index=False)
        .agg(avg_value=(value_col, "mean"))
        .sort_values(["year_month", "year"])
    )
    pivot_df = monthly_avg_df.pivot(index="year_month", columns="year", values="avg_value").sort_index()
    pivot_df.index.name = "year_month"
    return pivot_df


def _season_avg_pivot(day_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    season_avg_df = day_df.groupby(["season_label", "year"], as_index=False).agg(avg_value=(value_col, "mean"))
    pivot_df = season_avg_df.pivot(index="season_label", columns="year", values="avg_value").reindex(season_order)
    pivot_df.index.name = "season"
    return pivot_df


def _weather_avg_df(day_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    weather_avg_df = (
        day_df.groupby("weather_label", as_index=False)
        .agg(avg_value=(value_col, "mean"))
        .sort_values("avg_value", ascending=False)
    )
    return weather_avg_df


def _hour_weekday_pivot(hour_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_df = (
        hour_df.pivot_table(index="hr", columns="weekday_label", values=value_col, aggfunc="mean")
        .reindex(columns=weekday_order)
        .sort_index()
    )
    heatmap_df.index.name = "hr"
    return heatmap_df


def _hourly_workday_pivot(hour_df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_workday_df = (
        hour_df.groupby(["hr", "workingday"], as_index=False)
        .agg(avg_value=(value_col, "mean"))
        .sort_values(["workingday", "hr"])
    )
    hourly_workday_df["workingday_label"] = hourly_workday_df["workingday"].map(
        {0: "Weekend/Holiday", 1: "Workingday"}
    )
    pivot_df = hourly_workday_df.pivot(index="hr", columns="workingday_label", values="avg_value").sort_index()
    pivot_df.index.name = "hr"

    peak_hours_df = (
        hourly_workday_df.sort_values("avg_value", ascending=False)
        .groupby("workingday_label", as_index=False)
        .head(3)
        .sort_values(["workingday_label", "avg_value"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return pivot_df, peak_hours_df[["workingday_label", "hr", "avg_value"]]


def _time_group_pivot(hour_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    time_bins = [-0.1, 5, 10, 14, 18, 23]
    time_labels = [
        "Dini Hari (0-5)",
        "Pagi (6-10)",
        "Siang (11-14)",
        "Sore (15-18)",
        "Malam (19-23)",
    ]
    time_group_df = hour_df.copy()
    time_group_df["time_group"] = pd.cut(
        time_group_df["hr"],
        bins=time_bins,
        labels=time_labels,
        include_lowest=True,
    )
    time_group_df["workingday_label"] = time_group_df["workingday"].map(
        {0: "Weekend/Holiday", 1: "Workingday"}
    )
    avg_df = (
        time_group_df.groupby(["time_group", "workingday_label"], as_index=False)
        .agg(avg_value=(value_col, "mean"))
        .sort_values(["time_group", "workingday_label"])
    )
    pivot_df = avg_df.pivot(index="time_group", columns="workingday_label", values="avg_value").reindex(time_labels)
    pivot_df.index.name = "time_group"
    return pivot_df


def main() -> None:
    st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")
    st.title("Bike Sharing Dashboard")

    day_df, hour_df = _load_data()

    with st.sidebar:
        st.header("Filter")

        user_type = st.selectbox("Tipe pengguna", options=["Total", "Casual", "Registered"])
        user_type_to_col = {"Total": "cnt", "Casual": "casual", "Registered": "registered"}
        value_col = user_type_to_col[user_type]
        value_label = {"cnt": "Total", "casual": "Casual", "registered": "Registered"}[value_col]

        enable_date_filter = st.checkbox("Filter tanggal", value=False)
        min_date = day_df["dteday"].min().date()
        max_date = day_df["dteday"].max().date()
        start_date = min_date
        end_date = max_date
        if enable_date_filter:
            try:
                picked = st.date_input(
                    "Rentang tanggal",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(picked, tuple) and len(picked) == 2:
                    start_date, end_date = picked
                else:
                    start_date = picked
                    end_date = picked
            except Exception:
                st.warning("Rentang tanggal tidak valid. Menampilkan semua tanggal.")
                start_date = min_date
                end_date = max_date

        years = sorted(day_df["year"].dropna().unique().tolist())
        selected_years = st.multiselect("Tahun", options=years, default=years)

        seasons = ["Spring", "Summer", "Fall", "Winter"]
        selected_seasons = st.multiselect("Musim", options=seasons, default=seasons)

        weathers = [
            "Clear/Partly Cloudy",
            "Mist/Cloudy",
            "Light Snow/Rain",
            "Heavy Rain/Ice",
        ]
        selected_weathers = st.multiselect("Kondisi Cuaca", options=weathers, default=weathers[:3])

        temp_min = float(np.floor(day_df["temp_c"].min()))
        temp_max = float(np.ceil(day_df["temp_c"].max()))
        selected_temp = st.slider("Rentang Suhu (°C)", min_value=temp_min, max_value=temp_max, value=(temp_min, temp_max))

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    day_date_mask = day_df["dteday"].between(start_ts, end_ts, inclusive="both") if enable_date_filter else True
    hour_date_mask = hour_df["dteday"].between(start_ts, end_ts, inclusive="both") if enable_date_filter else True

    filtered_day_df = day_df[
        day_date_mask
        & day_df["year"].isin(selected_years)
        & day_df["season_label"].isin(selected_seasons)
        & day_df["weather_label"].isin(selected_weathers)
        & day_df["temp_c"].between(selected_temp[0], selected_temp[1], inclusive="both")
    ].copy()

    filtered_hour_df = hour_df[
        hour_date_mask
        & hour_df["year"].isin(selected_years)
        & hour_df["season_label"].isin(selected_seasons)
        & hour_df["weather_label"].isin(selected_weathers)
        & hour_df["temp_c"].between(selected_temp[0], selected_temp[1], inclusive="both")
    ].copy()

    if filtered_day_df.empty:
        st.warning("Data kosong untuk filter yang dipilih. Ubah filter di sidebar.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Total Penyewaan ({value_label})", f"{int(filtered_day_df[value_col].sum()):,}")
    col2.metric(f"Rata-rata Harian ({value_label})", f"{filtered_day_df[value_col].mean():.0f}")
    col3.metric("Jumlah Hari Teramati", f"{int(filtered_day_df.shape[0]):,}")

    st.subheader("Pertanyaan 1: Tren Bulanan & Musiman")
    c1, c2 = st.columns(2)
    with c1:
        monthly_pivot = _monthly_avg_pivot(filtered_day_df, value_col=value_col)
        st.line_chart(monthly_pivot)
    with c2:
        season_pivot = _season_avg_pivot(filtered_day_df, value_col=value_col)
        st.bar_chart(season_pivot)

    st.subheader("Pertanyaan 2: Pengaruh Cuaca & Suhu")
    c3, c4 = st.columns(2)
    with c3:
        weather_avg_df = _weather_avg_df(filtered_day_df, value_col=value_col).set_index("weather_label")
        st.bar_chart(weather_avg_df)
    with c4:
        scatter_df = filtered_day_df[["temp_c", value_col]].sort_values("temp_c")
        st.scatter_chart(scatter_df, x="temp_c", y=value_col)

    st.subheader("Analisis Tambahan: Pola Jam vs Hari")
    heatmap_df = _hour_weekday_pivot(filtered_hour_df, value_col=value_col)
    st.dataframe(heatmap_df, use_container_width=True)

    workday_pivot, peak_hours_df = _hourly_workday_pivot(filtered_hour_df, value_col=value_col)
    st.line_chart(workday_pivot)
    st.dataframe(peak_hours_df, use_container_width=True)

    st.subheader("Analisis Lanjutan: Pengelompokan Waktu (Manual Grouping)")
    time_group_pivot = _time_group_pivot(filtered_hour_df, value_col=value_col)
    st.bar_chart(time_group_pivot)

    st.subheader("Data (Preview)")
    st.dataframe(filtered_day_df.sort_values("dteday").tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
