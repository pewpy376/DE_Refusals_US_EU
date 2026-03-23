from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


def classify_barrier(section_name: str, stmt_text: str) -> str:
    text = f"{section_name} {stmt_text}".upper()
    has_403 = "403" in text
    has_402 = "402" in text

    if has_403 and has_402:
        return "BOTH"
    if has_403:
        return "TBT"
    if has_402:
        return "SPS"
    return "OTHER"


def build_unique_order_id(df: pd.DataFrame) -> pd.Series:
    id_cols = ["ENTRY_NUM", "RFRNC_DOC_ID", "LINE_NUM", "LINE_SFX_ID"]
    available = [c for c in id_cols if c in df.columns]

    if not available:
        return pd.Series([f"ROW_{i}" for i in df.index], index=df.index)

    order_id = (
        df[available]
        .fillna("")
        .astype(str)
        .apply(lambda row: "|".join(v.strip() for v in row.values), axis=1)
        .str.strip("|")
    )
    return order_id.mask(order_id.eq(""), "ROW_" + df.index.astype(str))


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    refusal_1 = pd.read_csv(base_dir / "REFUSAL_ENTRY_2019_2023.csv", encoding="latin-1", dtype=str)
    refusal_2 = pd.read_csv(base_dir / "REFUSAL_ENTRY_2024-Feb2026.csv", encoding="latin-1", dtype=str)
    refusal = pd.concat([refusal_1, refusal_2], ignore_index=True)

    charges_1 = pd.read_csv(base_dir / "ACT_SECTION_CHARGES_1923.csv", encoding="latin-1", dtype=str)
    charges_2 = pd.read_csv(base_dir / "ACT_SECTION_CHARGES_2426.csv", encoding="latin-1", dtype=str)
    charges = pd.concat([charges_1, charges_2], ignore_index=True)

    # Keep a single row per ASC_ID to avoid duplicate joins.
    charges["ASC_ID"] = charges["ASC_ID"].astype(str).str.strip()
    charges = charges.dropna(subset=["ASC_ID"]).drop_duplicates(subset=["ASC_ID"], keep="first")

    return refusal, charges


def prepare_data(refusal: pd.DataFrame, charges: pd.DataFrame, countries: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    filtered_refusal = refusal[refusal["ISO_CNTRY_CODE"].isin(countries)].copy()
    filtered_refusal["REFUSAL_DATE"] = pd.to_datetime(filtered_refusal["REFUSAL_DATE"], errors="coerce")
    filtered_refusal["UNIQUE_ORDER_ID"] = build_unique_order_id(filtered_refusal)
    counts = filtered_refusal["ISO_CNTRY_CODE"].value_counts().reindex(countries, fill_value=0)

    filtered_refusal["REFUSAL_CHARGES_LIST"] = (
        filtered_refusal["REFUSAL_CHARGES"].fillna("").astype(str).str.split(",")
    )
    exploded = filtered_refusal.explode("REFUSAL_CHARGES_LIST")

    exploded["REFUSAL_CHARGES_LIST"] = exploded["REFUSAL_CHARGES_LIST"].astype(str).str.strip()
    exploded = exploded[exploded["REFUSAL_CHARGES_LIST"] != ""]

    exploded = exploded.merge(
        charges,
        left_on="REFUSAL_CHARGES_LIST",
        right_on="ASC_ID",
        how="left",
    )
    return exploded, counts


def export_final_output(exploded: pd.DataFrame, base_dir: Path) -> Path:
    country_names = {"VN": "Vietnam", "EC": "Ecuador", "IN": "India"}

    final_df = exploded.copy()
    final_df["Barrier Classification"] = final_df.apply(
        lambda row: classify_barrier(row.get("SCTN_NAME", ""), row.get("CHRG_STMNT_TEXT", "")), axis=1
    )

    final_df["Importing Market"] = "United States"
    final_df["Exporting country"] = final_df["ISO_CNTRY_CODE"].map(country_names).fillna(final_df["ISO_CNTRY_CODE"])
    final_df["Refusal date - formating as MM-YYYY"] = final_df["REFUSAL_DATE"].dt.strftime("%m-%Y")
    final_df["Specific Reason of Rejection"] = final_df["CHRG_STMNT_TEXT"].fillna("").astype(str).str.strip()
    final_df["Specific Reason of Rejection"] = final_df["Specific Reason of Rejection"].mask(
        final_df["Specific Reason of Rejection"].eq(""),
        final_df["CHRG_CODE"].fillna("Unknown").astype(str),
    )
    final_df["Unique Order ID"] = final_df["UNIQUE_ORDER_ID"].astype(str)

    both_rows = final_df[final_df["Barrier Classification"] == "BOTH"].copy()
    if not both_rows.empty:
        tbt_rows = both_rows.copy()
        tbt_rows["Barrier Classification"] = "TBT"
        sps_rows = both_rows.copy()
        sps_rows["Barrier Classification"] = "SPS"
        final_df = pd.concat([final_df[final_df["Barrier Classification"] != "BOTH"], tbt_rows, sps_rows], ignore_index=True)

    final_df = final_df[final_df["Barrier Classification"].isin(["TBT", "SPS"])].copy()
    final_df = final_df[
        [
            "Importing Market",
            "Exporting country",
            "Refusal date - formating as MM-YYYY",
            "Barrier Classification",
            "Specific Reason of Rejection",
            "Unique Order ID",
        ]
    ]

    final_df = final_df.dropna(subset=["Refusal date - formating as MM-YYYY"])
    final_df = final_df.drop_duplicates(
        ["Unique Order ID", "Barrier Classification", "Specific Reason of Rejection"]
    ).sort_values(
        ["Exporting country", "Refusal date - formating as MM-YYYY", "Unique Order ID", "Barrier Classification"]
    )

    output_path = base_dir / "FDA_final_output.xlsx"
    final_df.to_excel(output_path, index=False)
    return output_path


def print_top_reasons(exploded: pd.DataFrame, charges: pd.DataFrame, countries: list[str], top_n: int = 5) -> None:
    for country in countries:
        print(f"--- Top Charges for {country} ---")
        country_data = exploded[exploded["ISO_CNTRY_CODE"] == country]
        top_charges = country_data["CHRG_CODE"].fillna("Unknown").value_counts().head(top_n)

        for charge_code, count in top_charges.items():
            if charge_code == "Unknown":
                desc_text = "Unknown"
            else:
                desc = charges.loc[charges["CHRG_CODE"] == charge_code, "CHRG_STMNT_TEXT"].values
                desc_text = desc[0] if len(desc) > 0 else "Unknown"
            print(f"[{charge_code}] - {count} times")
            print(f"  Desc: {str(desc_text)[:150]}...")
        print()


def create_visualizations(exploded: pd.DataFrame, counts: pd.Series, countries: list[str], base_dir: Path) -> None:
    country_names = {"VN": "Vietnam", "EC": "Ecuador", "IN": "India"}
    labels = [country_names.get(c, c) for c in countries]

    plt.style.use("ggplot")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(labels, counts.values, color=["#2c7fb8", "#f03b20", "#31a354"])
    ax1.set_title("Total FDA Refusals by Country")
    ax1.set_xlabel("Country")
    ax1.set_ylabel("Number of Refusals")
    for idx, val in enumerate(counts.values):
        ax1.text(idx, int(val) + 5, str(int(val)), ha="center", fontsize=10)
    fig1.tight_layout()
    fig1.savefig(base_dir / "fda_refusals_by_country.png", dpi=200)
    plt.close(fig1)

    top_n = 5
    fig2, axes = plt.subplots(1, len(countries), figsize=(16, 5), sharey=False)
    if len(countries) == 1:
        axes = [axes]

    for i, country in enumerate(countries):
        country_data = exploded[exploded["ISO_CNTRY_CODE"] == country]
        top_codes = country_data["CHRG_CODE"].fillna("Unknown").value_counts().head(top_n)
        axes[i].barh(top_codes.index.astype(str), top_codes.values, color="#4c78a8")
        axes[i].invert_yaxis()
        axes[i].set_title(country_names.get(country, country))
        axes[i].set_xlabel("Count")

    for ax in axes:
        ax.set_ylabel("Charge Code")
    fig2.suptitle("Top 5 FDA Refusal Charges by Country", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(base_dir / "fda_top_charges_by_country.png", dpi=200)
    plt.close(fig2)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    countries = ["VN", "EC", "IN"]

    refusal, charges = load_data(base_dir)
    exploded, counts = prepare_data(refusal, charges, countries)

    print("Total Refusals:")
    print(counts)
    print()

    print_top_reasons(exploded, charges, countries, top_n=5)
    create_visualizations(exploded, counts, countries, base_dir)
    output_file = export_final_output(exploded, base_dir)

    print("Saved plots:")
    print(f"- {base_dir / 'fda_refusals_by_country.png'}")
    print(f"- {base_dir / 'fda_top_charges_by_country.png'}")
    print(f"Saved Excel output:\n- {output_file}")


if __name__ == "__main__":
    main()