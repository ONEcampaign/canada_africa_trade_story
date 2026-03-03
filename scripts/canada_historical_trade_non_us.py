"""
Canada historical trade data pipeline.

Pulls bilateral trade data from BACI (HS12 revision), deflates values to
constant 2023 CAD, and produces several CSV outputs broken down by region,
trade category, and partner country.

Output files (written to Paths.output):
  canada_africa_trade_by_country_category_cad.csv  — Africa partners × category, absolute values
  canada_africa_trade_by_country_category_pct.csv  — same, % of global flow
  canada_trade_by_region_category_cad.csv          — all regions × category, absolute values
  canada_trade_by_region_category_pct.csv          — same, % of global flow
  canada_africa_trade_by_category.csv              — Africa totals by category, absolute values
  canada_trade_by_region_cad.csv                   — all regions, absolute values
  canada_trade_by_region_pct.csv                   — all regions, % of global flow
"""

import json

import pandas as pd
from bblocks.data_importers import BACI
from bblocks.places import resolve_places
from pydeflate import imf_gdp_deflate, set_pydeflate_path

from scripts.config import Paths
from scripts.helpers import fix_encoding, group_trade_data, rename_flows

HS_VERSION = "HS12"

YEAR_RANGE = [2015, 2024]

BASE_YEAR = 2024

# BACI country code for Canada and US
CANADA_CODE = 124

US_CODE = 842

# Region labels used consistently across outputs
REGION_ORDER = [
    "North America",
    "Asia",
    "Europe",
    "Central, South America & Caribbean",
    "Africa",
    "Oceania",
]

# Commodity category labels used consistently across outputs
CATEGORY_ORDER = [
    "Manufactured goods",
    "Agri-food products",
    "Minerals & fuels",
    "Precious metals & stones",
]

# Flow label mapping applied to all outputs
FLOW_MAP = {"Imports": "Canada imports", "Exports": "Canada exports"}


def load_prepare_data() -> pd.DataFrame:
    """
    Load raw BACI trade data and return Canada-specific bilateral flows by category.

    Steps:
    1. Fetch HS12 trade data and country code lookup from BACI.
    2. Map each 6-digit HS product code to a broad trade category via the
       first 2 digits (HS2 chapter), using the mapping in hs_categories.json.
    3. Filter to rows where Canada is importer or exporter.
    4. Aggregate trade value by year / importer / exporter / category.
    5. Join ISO3 codes and country names for both importer and exporter.

    Returns
    -------
    DataFrame with columns:
        year, category, value, importer_iso3, importer_name,
        exporter_iso3, exporter_name
    """
    baci = BACI()
    df_raw = baci.get_data(hs_version=HS_VERSION)
    country_codes = baci.get_country_codes(hs_version=HS_VERSION)

    with open(Paths.hs_categories, "r") as f:
        hs_map = json.load(f)
    # Map HS2 chapter string (e.g. "09") → broad category label
    hs_to_category = {hs2: entry["one_section"] for hs2, entry in hs_map.items()}

    df = (

    df_raw
    # Filter early to shrink the dataset massively (CRITICAL)
    .loc[lambda d: d["year"].between (YEAR_RANGE[0], YEAR_RANGE[1])]
    .loc[lambda d: (d["importer_code"].eq(CANADA_CODE)) | (d["exporter_code"].eq(CANADA_CODE))]
    # Compute HS2 category using numeric operations (avoid stringifying 140M rows)
    .assign(
        hs2=lambda d: d["product_code"].astype("int64") // 10000,
        category=lambda d: d["hs2"].astype(str).str.zfill(2).map(hs_to_category),
    )
    .drop(columns=["hs2"])
    # Aggregate to Canada bilateral flows by category
    .groupby(["year", "importer_code", "exporter_code", "category"], as_index=False)["value"]
    .sum()
    # Attach ISO3 codes and country names for both trade partners
    .merge(
        country_codes.rename(
            columns={
                "country_code": "importer_code",
                "iso3_code": "importer_iso3",
                "country_name": "importer_name",
            }
        ),
        on="importer_code",
        how="left",
    )
    .merge(
        country_codes.rename(
            columns={
                "country_code": "exporter_code",
                "iso3_code": "exporter_iso3",
                "country_name": "exporter_name",
            }
        ),
        on="exporter_code",
        how="left",
    )
    .drop(columns=["importer_code", "exporter_code"])
    )
    return df


def deflate_trade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert trade values from current USD to constant 2023 CAD.

    Uses IMF GDP deflators via pydeflate. The source currency is USD (BACI
    values are reported in thousands of current USD) and the target is CAD,
    rebased to 2023.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'value' column in current USD thousands.

    Returns
    -------
    DataFrame with 'value' replaced by constant 2023 CAD thousands.
    """
    set_pydeflate_path(Paths.pydeflate)

    df["iso3"] = "CAN"
    df = imf_gdp_deflate(
        df,
        base_year=BASE_YEAR,
        source_currency="USA",
        target_currency="CAN",
        id_column="iso3",
        year_column="year",
        value_column="value",
        target_value_column="value",
    ).drop(columns="iso3")

    return df


def reshape_trade_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot bilateral flows into a partner-centric view with Imports and Exports columns.

    Splits the data into Canada-as-exporter and Canada-as-importer, then
    outer-joins them on (year, partner_iso3, partner, category) so each row
    represents one trade relationship in one year.

    Parameters
    ----------
    df : DataFrame
        Output of deflate_trade(); contains importer_iso3, exporter_iso3,
        importer_name, exporter_name, year, category, value.

    Returns
    -------
    DataFrame with columns:
        year, partner_iso3, partner, category, Exports, Imports,
        unit ('thousand CAD, base 2023')
    """
    exports_df = (
        df[df["exporter_iso3"] == "CAN"]
        .rename(
            columns={
                "value": "Exports",
                "importer_iso3": "partner_iso3",
                "importer_name": "partner",
            }
        )
        .drop(columns=["exporter_iso3", "exporter_name"])
    )

    imports_df = (
        df[df["importer_iso3"] == "CAN"]
        .rename(
            columns={
                "value": "Imports",
                "exporter_iso3": "partner_iso3",
                "exporter_name": "partner",
            }
        )
        .drop(columns=["importer_iso3", "importer_name"])
    )

    merged = pd.merge(
        exports_df,
        imports_df,
        how="outer",
        on=["year", "partner_iso3", "partner", "category"],
    ).assign(
        Imports=lambda d: d["Imports"].fillna(0),
        Exports=lambda d: d["Exports"].fillna(0),
        unit="thousand CAD, base 2023",
    )

    return merged


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'region' column based on partner ISO3 code.

    Uses bblocks resolve_places for standard region assignment, then applies
    manual overrides:
    - 'Americas' is split into 'North America' (BMU, USA, SPM, GRL) and
      'Central, South America & Caribbean' (everything else in Americas).
    - 'S19' (Asia n.e.s.) is mapped to 'Asia'.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'partner_iso3' column.

    Returns
    -------
    DataFrame with an added 'region' column.
    """
    df = df.copy()

    df["region"] = resolve_places(
        df["partner_iso3"], from_type="iso3_code", to_type="region", not_found="ignore"
    )

    # Canada's non-continental North American partners that resolve_places
    # groups under 'Americas' but belong in the North America bucket
    north_america = ["BMU", "USA", "SPM", "GRL"]
    df.loc[df["partner_iso3"].isin(north_america), "region"] = "North America"
    df.loc[
        (df["region"] == "Americas") & ~df["partner_iso3"].isin(north_america), "region"
    ] = "Central, South America & Caribbean"

    # 'S19' is the BACI code for unspecified Asian territories
    df.loc[df["partner_iso3"] == "S19", "region"] = "Asia"

    return df

def add_us_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column that splits partners into 'USA' and 'Non-USA' buckets.
    """
    df = df.copy()
    df["us_bucket"] = df["partner_iso3"].apply(lambda x: "USA" if x == "USA" else "Non-USA")
    return df

def main():
    # ------------------------------------------------------------------ #
    # Build the core dataset                                               #
    # ------------------------------------------------------------------ #
    df_raw = load_prepare_data()
    df_deflated = deflate_trade(df_raw)
    df_trade = reshape_trade_data(df_deflated)
    df_geo = add_region(df_trade)
    df_geo = add_us_bucket(df_geo)

    # Some country names from BACI are mis-encoded; fix before writing CSVs
    df_geo["partner"] = df_geo["partner"].apply(fix_encoding)

    # ------------------------------------------------------------------ #
    # Africa: by partner country and trade category                        #
    # ------------------------------------------------------------------ #
    df_africa_by_country_cat_cad, df_africa_by_country_cat_pct = group_trade_data(
        df_geo.query("region == 'Africa'"),
        groupby_cols=["year", "partner", "category"],
        pivot_column="category",
        cols_order=CATEGORY_ORDER,
    )

    rename_flows(df_africa_by_country_cat_cad, FLOW_MAP).to_csv(
        Paths.output / "canada_africa_trade_by_country_category_cad.csv", index=False
    )
    rename_flows(df_africa_by_country_cat_pct, FLOW_MAP).to_csv(
        Paths.output / "canada_africa_trade_by_country_category_pct.csv", index=False
    )

    # ------------------------------------------------------------------ #
    # All regions: by region and trade category                            #
    # ------------------------------------------------------------------ #
    df_by_reg_cat_cad, df_by_reg_cat_pct = group_trade_data(
        df_geo,
        groupby_cols=["year", "region", "category"],
        pivot_column="category",
        cols_order=CATEGORY_ORDER,
    )

    rename_flows(df_by_reg_cat_cad, FLOW_MAP).to_csv(
        Paths.output / "canada_trade_by_region_category_cad.csv", index=False
    )
    rename_flows(df_by_reg_cat_pct, FLOW_MAP).to_csv(
        Paths.output / "canada_trade_by_region_category_pct.csv", index=False
    )

    # Africa slice of the above (absolute values only)
    df_by_reg_cat_cad.query("region == 'Africa'").to_csv(
        Paths.output / "canada_africa_trade_by_category.csv", index=False
    )

    # ------------------------------------------------------------------ #
    # All regions: totals by region                                        #
    # ------------------------------------------------------------------ #
    df_by_region_cad, df_by_region_pct = group_trade_data(
        df_geo,
        groupby_cols=["year", "region"],
        pivot_column="region",
        cols_order=REGION_ORDER,
    )

    rename_flows(df_by_region_cad, FLOW_MAP).to_csv(
        Paths.output / "canada_trade_by_region_cad.csv", index=False
    )
    rename_flows(df_by_region_pct, FLOW_MAP).to_csv(
        Paths.output / "canada_trade_by_region_pct.csv", index=False
    )
    df_us_vs_nonus_cad, df_us_vs_nonus_pct = group_trade_data(
        df_geo,
        groupby_cols=["year", "us_bucket"],
        pivot_column="us_bucket",
        cols_order=["USA", "Non-USA"],
    )
    rename_flows(df_us_vs_nonus_cad, FLOW_MAP).to_csv(
        Paths.output / "canada_trade_by_us_nonus_cad.csv", index=False
    )
    rename_flows(df_us_vs_nonus_pct, FLOW_MAP).to_csv(
        Paths.output / "canada_trade_by_us_nonus_pct.csv", index=False
    )


if __name__ == "__main__":
    main()
