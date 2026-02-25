"""
Generic utility functions shared across pipeline scripts.
"""

import pandas as pd


def group_trade_data(
    df: pd.DataFrame,
    groupby_cols: list[str],
    pivot_column: str | None = None,
    cols_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate trade data, compute global percentage shares, and pivot.

    For each combination of groupby_cols, Imports and Exports are summed.
    Percentage shares are calculated relative to total global flow per year
    (i.e. each value as a % of the worldwide total for that flow direction
    and year, across all rows passed in).

    Parameters
    ----------
    df : DataFrame
        Must contain 'Imports', 'Exports', and 'year' columns.
    groupby_cols : list of str
        Columns to group by before pivoting, e.g. ['year', 'region'].
    pivot_column : str or None
        Column whose values become the output columns after pivoting,
        e.g. 'region' or 'category'. If None, no pivot is applied.
    cols_order : list of str or None
        Desired column order in the pivoted output. Columns not present
        in the data are silently ignored.

    Returns
    -------
    df_value : DataFrame
        Pivoted table with absolute trade values.
    df_pct : DataFrame
        Pivoted table with each value as a % of the global flow total.
    """
    df_grouped = df.groupby(groupby_cols)[["Exports", "Imports"]].sum().reset_index()

    df_melted = df_grouped.melt(
        id_vars=groupby_cols,
        value_vars=["Imports", "Exports"],
        var_name="flow",
        value_name="value",
    )

    # Percentage share of total global flow for each direction and year
    df_melted["value_pct"] = df_melted.groupby(["year", "flow"])["value"].transform(
        lambda x: x / x.sum() * 100
    )

    pivot_index = [col for col in groupby_cols if col != pivot_column] + ["flow"]

    df_value = df_melted.pivot_table(
        index=pivot_index, columns=pivot_column, values="value", aggfunc="sum"
    )
    df_pct = df_melted.pivot_table(
        index=pivot_index, columns=pivot_column, values="value_pct", aggfunc="sum"
    )

    if cols_order is not None:
        existing_cols = [col for col in cols_order if col in df_value.columns]
        df_value = df_value[existing_cols]
        df_pct = df_pct[existing_cols]

    df_value = df_value.reset_index()
    df_pct = df_pct.reset_index()
    df_value.columns.name = None
    df_pct.columns.name = None

    return df_value, df_pct


def rename_flows(df: pd.DataFrame, flow_map: dict) -> pd.DataFrame:
    """
    Rename values in the 'flow' column using the provided mapping.

    Unmapped values are left unchanged.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'flow' column.
    flow_map : dict
        Mapping from old flow labels to new ones,
        e.g. {'Imports': 'Canada imports', 'Exports': 'Canada exports'}.
    """
    df = df.copy()
    df["flow"] = df["flow"].map(flow_map).fillna(df["flow"])
    return df


def fix_encoding(s: str) -> str:
    """
    Fix country names that were read as latin-1 but are encoded as UTF-8.

    Some country names sourced from BACI arrive with mojibake (e.g. "CÃ´te
    d'Ivoire" instead of "Côte d'Ivoire"). This re-encodes them correctly.
    """
    return s.encode("latin1").decode("utf-8")
