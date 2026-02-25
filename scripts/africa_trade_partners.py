"""
Africa trade partners comparison.

Compares bilateral trade flows between Africa (aggregate) and key global
partners — EU27, ASEAN, and selected individual countries — for 2005 and 2025.

Output (Paths.output / "africa_trade_partners.csv"):
  country  — trading bloc or country name (EU27 / ASEAN / individual)
  flow     — direction from the partner's perspective (Exports / Imports)
  2005     — trade value in 2005
  2024     — trade value in 2024
  unit     — 'thousand USD, constant 2023' or '% of total trade'
"""

import pandas as pd
from bblocks.data_importers import BACI
from bblocks.places import filter_african_countries
from pydeflate import imf_gdp_deflate, set_pydeflate_path

from scripts.config import Paths

HS_VERSION = "HS02"

TARGET_YEARS = [2005, 2024]

BASE_YEAR = 2024

EU_27_CODES = [
    "AUT",  # Austria
    "BEL",  # Belgium
    "BGR",  # Bulgaria
    "HRV",  # Croatia
    "CYP",  # Cyprus
    "CZE",  # Czechia
    "DNK",  # Denmark
    "EST",  # Estonia
    "FIN",  # Finland
    "FRA",  # France
    "DEU",  # Germany
    "GRC",  # Greece
    "HUN",  # Hungary
    "IRL",  # Ireland
    "ITA",  # Italy
    "LVA",  # Latvia
    "LTU",  # Lithuania
    "LUX",  # Luxembourg
    "MLT",  # Malta
    "NLD",  # Netherlands
    "POL",  # Poland
    "PRT",  # Portugal
    "ROU",  # Romania
    "SVK",  # Slovakia
    "SVN",  # Slovenia
    "ESP",  # Spain
    "SWE",  # Sweden
]

ASEAN_CODES = [
    "BRN",  # Brunei
    "KHM",  # Cambodia
    "IDN",  # Indonesia
    "LAO",  # Laos
    "MYS",  # Malaysia
    "MMR",  # Myanmar
    "PHL",  # Philippines
    "SGP",  # Singapore
    "THA",  # Thailand
    "VNM",  # Vietnam
]

OTHER_CODES = [
    "CAN",  # Canada
    "CHN",  # China
    "IND",  # India
    "GBR",  # United Kingdom
    "USA",  # United States
]

# Display names for individual countries in OTHER_CODES
OTHER_NAMES = {
    "CAN": "Canada",
    "CHN": "China",
    "IND": "India",
    "GBR": "United Kingdom",
    "USA": "United States",
}

ALL_PARTNER_CODES = set(EU_27_CODES + ASEAN_CODES + OTHER_CODES)


def load_format_data() -> pd.DataFrame:
    """
    Load BACI HS02 data, aggregate to yearly bilateral totals, and attach ISO3 codes.

    Returns
    -------
    DataFrame with columns: year, importer_iso3, exporter_iso3, value
        where value is in thousands of current USD.
    """
    baci = BACI()
    df_raw = baci.get_data(hs_version="HS02")
    country_codes = baci.get_country_codes()[["country_code", "iso3_code"]]

    return (
        df_raw
        .groupby(["year", "importer_code", "exporter_code"])["value"]
        .sum()
        .reset_index()
        .merge(
            country_codes.rename(columns={"country_code": "importer_code", "iso3_code": "importer_iso3"}),
            on="importer_code",
            how="left",
        )
        .merge(
            country_codes.rename(columns={"country_code": "exporter_code", "iso3_code": "exporter_iso3"}),
            on="exporter_code",
            how="left",
        )
        .drop(columns=["importer_code", "exporter_code"])
    )


def deflate_trade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deflate trade values from current USD to constant CAD using each partner's deflator.

    Must be called after filter_africa_partner_trade so that 'partner_iso3' is
    available. Each row is deflated using the GDP deflator of the non-African
    trading partner, reflecting inflation from that country's economic perspective.

    Parameters
    ----------
    df : DataFrame
        Must contain 'value' (current USD thousands) and 'partner_iso3' columns.

    Returns
    -------
    DataFrame with 'value' replaced by constant BASE_YEAR CAD thousands.
    """
    set_pydeflate_path(Paths.pydeflate)

    return (
        imf_gdp_deflate(
            df,
            base_year=BASE_YEAR,
            source_currency="USA",
            target_currency="CAN",
            id_column="partner_iso3",
            year_column="year",
            value_column="value",
            target_value_column="value",
        )
    )


def filter_africa_partner_trade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where one side is an African country and the other is a tracked partner.

    Determines the African ISO3 codes from the data itself using
    filter_african_countries, then separates the data into two flow directions:
    - Exports: the tracked partner is the exporter, Africa is the importer
    - Imports: the tracked partner is the importer, Africa is the exporter

    Parameters
    ----------
    df : DataFrame
        Must contain 'importer_iso3', 'exporter_iso3', and 'value' columns.

    Returns
    -------
    DataFrame with added columns:
        partner_iso3 — ISO3 code of the non-African trading partner
        flow         — 'Exports' or 'Imports' from the partner's perspective
    """
    # Derive the full set of African ISO3 codes present in the dataset
    all_codes = pd.concat([df["importer_iso3"], df["exporter_iso3"]]).dropna().unique().tolist()
    african_iso3 = set(
        filter_african_countries(all_codes, from_type="iso3_code", not_found="ignore")
    )

    # Partner exports to Africa: partner is exporter, African country is importer
    exports_to_africa = (
        df[df["importer_iso3"].isin(african_iso3) & df["exporter_iso3"].isin(ALL_PARTNER_CODES)]
        .assign(flow="Exports", partner_iso3=lambda d: d["exporter_iso3"])
    )

    # Partner imports from Africa: partner is importer, African country is exporter
    imports_from_africa = (
        df[df["exporter_iso3"].isin(african_iso3) & df["importer_iso3"].isin(ALL_PARTNER_CODES)]
        .assign(flow="Imports", partner_iso3=lambda d: d["importer_iso3"])
    )

    return pd.concat([exports_to_africa, imports_from_africa], ignore_index=True)


def assign_partner_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'country' column that groups EU27 and ASEAN members and names others.

    EU27 members → 'EU27'
    ASEAN members → 'ASEAN'
    Other tracked countries → display name from OTHER_NAMES (falls back to ISO3)

    Parameters
    ----------
    df : DataFrame
        Must contain a 'partner_iso3' column.

    Returns
    -------
    DataFrame with an added 'country' column.
    """
    eu27_set = set(EU_27_CODES)
    asean_set = set(ASEAN_CODES)

    def _group(iso3: str) -> str:
        if iso3 in eu27_set:
            return "EU27"
        if iso3 in asean_set:
            return "ASEAN"
        return OTHER_NAMES.get(iso3, iso3)

    return df.assign(country=lambda d: d["partner_iso3"].map(_group))


def aggregate_by_group_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum trade values by country group, flow direction, and year.

    Also filters to TARGET_YEARS only.

    Parameters
    ----------
    df : DataFrame
        Must contain 'country', 'flow', 'year', and 'value' columns.

    Returns
    -------
    DataFrame with one row per (country, flow, year) combination.
    """
    return (
        df[df["year"].isin(TARGET_YEARS)]
        .groupby(["country", "flow", "year"])["value"]
        .sum()
        .reset_index()
    )


def compute_partner_totals(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each tracked partner's total trade with the whole world per year and flow.

    This is used as the denominator for the percentage calculation, so that
    e.g. UK Exports % = UK exports to Africa / UK total exports worldwide.

    Uses the same deflation and grouping logic as the Africa-specific pipeline.

    Parameters
    ----------
    df_raw : DataFrame
        Full bilateral BACI data from load_format_data(); unfiltered.

    Returns
    -------
    DataFrame with columns: country, flow, year, total_value
        where total_value is total trade in CAD billion, constant BASE_YEAR.
    """
    # Tracked partner as exporter → their total exports to the whole world
    total_exports = (
        df_raw[df_raw["exporter_iso3"].isin(ALL_PARTNER_CODES)]
        .assign(flow="Exports", partner_iso3=lambda d: d["exporter_iso3"])
    )

    # Tracked partner as importer → their total imports from the whole world
    total_imports = (
        df_raw[df_raw["importer_iso3"].isin(ALL_PARTNER_CODES)]
        .assign(flow="Imports", partner_iso3=lambda d: d["importer_iso3"])
    )

    return (
        pd.concat([total_exports, total_imports], ignore_index=True)
        .pipe(deflate_trade)
        .pipe(assign_partner_group)
        .loc[lambda d: d["year"].isin(TARGET_YEARS)]
        .groupby(["country", "flow", "year"])["value"]
        .sum()
        .reset_index()
        .assign(total_value=lambda d: d["value"] / 1_000_000)
        .drop(columns="value")
    )


def build_wide_output(df: pd.DataFrame, df_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot years into columns and stack absolute-value and percentage-share rows.

    Percentage share is each partner's Africa trade as a share of their total
    worldwide trade in the same flow direction and year. For example, UK Exports
    % = UK exports to Africa / UK total exports to the world × 100.

    Parameters
    ----------
    df : DataFrame
        Output of aggregate_by_group_year(); columns: country, flow, year, value.
    df_totals : DataFrame
        Output of compute_partner_totals(); columns: country, flow, year, total_value.

    Returns
    -------
    DataFrame with columns: country, flow, <year1>, <year2>, unit.
    """
    df = df.copy()
    df["value_bn"] = df["value"] / 1_000_000  # thousands of CAD → billions of CAD
    df = df.merge(df_totals, on=["country", "flow", "year"], how="left")
    df["pct"] = df["value_bn"] / df["total_value"] * 100

    df_abs = (
        df.pivot_table(index=["country", "flow"], columns="year", values="value_bn")
        .reset_index()
        .assign(unit=f"CAD billion, constant {BASE_YEAR}")
    )
    df_abs.columns.name = None

    df_pct = (
        df.pivot_table(index=["country", "flow"], columns="year", values="pct")
        .reset_index()
        .assign(unit="% of partner's total trade")
    )
    df_pct.columns.name = None

    return (
        pd.concat([df_abs, df_pct], ignore_index=True)
        .sort_values(["country", "flow", "unit"])
        .reset_index(drop=True)
    )


def main():
    df_raw = load_format_data()

    # Africa-specific trade flows
    df_filtered = filter_africa_partner_trade(df_raw)
    df_deflated = deflate_trade(df_filtered)
    df_grouped = assign_partner_group(df_deflated)
    df_agg = aggregate_by_group_year(df_grouped)

    # Each partner's total worldwide trade (denominator for % calculation)
    df_totals = compute_partner_totals(df_raw)

    df_out = build_wide_output(df_agg, df_totals)

    df_out.to_csv(Paths.output / "africa_trade_partners.csv", index=False)


if __name__ == "__main__":
    main()