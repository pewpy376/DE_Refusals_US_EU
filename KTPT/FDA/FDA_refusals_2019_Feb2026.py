from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd


# ---------- Configuration ----------
START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2026-02-28")
TARGET_COUNTRIES = {"VN": "Vietnam", "IN": "India", "EC": "Ecuador"}

# Default patterns. Update only if your filenames are different.
REFUSAL_FILE_PATTERNS = [
    "REFUSAL_ENTRY_2019_2023.csv",
    "REFUSAL_ENTRY_2024-Feb2026.csv",
]
ACT_SECTION_FILE_PATTERNS = [
    "ACT_SECTION_CHARGES_1923*.csv",
    "ACT_SECTION_CHARGES_2426*.csv",
]

COUNTRY_NAMES = {
    "VN": "Vietnam",
    "IN": "India",
    "EC": "Ecuador",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def find_files(base_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base_dir.glob(pattern))
    # Keep insertion order while removing duplicates.
    return list(dict.fromkeys(files))


def read_table(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Exception | None = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=enc)
            print(f"Loaded {file_path.name} with encoding={enc}")
            return df
        except UnicodeDecodeError as err:
            last_err = err

    # If all strict attempts fail, replace undecodable bytes so processing can continue.
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8", encoding_errors="replace")
        print(f"Loaded {file_path.name} with encoding=utf-8 (invalid bytes replaced)")
        return df
    except Exception as err:
        last_err = err

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Could not decode {file_path.name} with tried encodings: {encodings_to_try}. Last error: {last_err}",
    )


def choose_refusal_files(base_dir: Path) -> list[Path]:
    explicit = [base_dir / "REFUSAL_ENTRY_2019_2023.csv", base_dir / "REFUSAL_ENTRY_2024-Feb2026.csv"]
    if all(p.exists() for p in explicit):
        return explicit
    found = find_files(base_dir, REFUSAL_FILE_PATTERNS)
    if not found:
        found = [p for pat in REFUSAL_FILE_PATTERNS for p in base_dir.parent.rglob(pat)]
    found = [p for p in found if "REFUSAL_ENTRY" in p.name.upper()]
    if not found:
        raise FileNotFoundError(
            "No refusal files found. Put refusal files in the same folder as this script, "
            "for example: REFUSAL_ENTRY_2019_2023.csv and REFUSAL_ENTRY_2024-Feb2026.csv"
        )
    return sorted(found)


def choose_act_file(base_dir: Path) -> Path:
    found = find_files(base_dir, ACT_SECTION_FILE_PATTERNS)
    if not found:
        found = [p for pat in ACT_SECTION_FILE_PATTERNS for p in base_dir.parent.rglob(pat)]
    if not found:
        raise FileNotFoundError(
            "No act section charges file found. Put a file with ASC_ID and charge text in this folder "
            "(csv or xlsx), for example: ACT_SECTION_CHARGES.csv"
        )
    return sorted(found)[0]


def extract_asc_ids(charge_text: str) -> list[str]:
    if pd.isna(charge_text):
        return []
    return re.findall(r"\d+", str(charge_text))


def classify_reason(section_name: str, stmt_text: str) -> str:
    text = f"{section_name} {stmt_text}".upper()

    # Strong statutory signal used in FDA refusal logic.
    if re.search(r"\b403\b", text):
        return "TBT"
    if re.search(r"\b402\b", text):
        return "SPS"

    tbt_keywords = [
        "MISBRANDING",
        "LABEL",
        "LABELING",
        "FALSE",
        "MISLEADING",
        "NUTRITION",
        "SPECIAL DIETARY",
        "MARKING",
        "PACKAGING",
        "BRAND",
    ]
    sps_keywords = [
        "ADULTERAT",
        "SALMONELLA",
        "LISTERIA",
        "VIBRIO",
        "E COLI",
        "E. COLI",
        "FILTH",
        "INSANITARY",
        "POISON",
        "VETERINARY",
        "ANTIBIOTIC",
        "DRUG",
        "PESTICIDE",
        "TOXIN",
        "CONTAMIN",
        "PARASITE",
        "DISEASE",
        "HACCP",
    ]

    has_tbt = any(k in text for k in tbt_keywords)
    has_sps = any(k in text for k in sps_keywords)
    if has_tbt and not has_sps:
        return "TBT"
    if has_sps and not has_tbt:
        return "SPS"
    if has_tbt and has_sps:
        return "BOTH"
    return "OTHER"


def build_order_id(df: pd.DataFrame) -> pd.Series:
    id_cols = [
        "ENTRY_NUM",
        "PFRNG_DOC_ID",
        "LINE_NUM",
        "LINE_SFX_ID",
    ]
    available = [c for c in id_cols if c in df.columns]
    if not available:
        return pd.Series([f"ROW_{i}" for i in df.index], index=df.index)

    id_part = (
        df[available]
        .fillna("")
        .astype(str)
        .apply(lambda r: "|".join(x.strip() for x in r.values), axis=1)
        .str.strip("|")
    )
    return id_part.mask(id_part.eq(""), "ROW_" + df.index.astype(str))


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    refusal_files = choose_refusal_files(base_dir)
    act_file = choose_act_file(base_dir)

    print("Refusal files:")
    for p in refusal_files:
        print(f"  - {p.name}")
    print(f"Act section file:\n  - {act_file.name}")

    # 1) Load data.
    refusals = pd.concat([normalize_columns(read_table(p)) for p in refusal_files], ignore_index=True)
    act = normalize_columns(read_table(act_file))

    required_refusal_cols = ["REFUSAL_DATE", "ISO_CNTRY_CODE", "REFUSAL_CHARGES"]
    missing_refusal = [c for c in required_refusal_cols if c not in refusals.columns]
    if missing_refusal:
        raise KeyError(f"Missing columns in refusal data: {missing_refusal}")

    required_act_cols = ["ASC_ID", "CHRG_STMNT_TEXT", "SCTN_NAME"]
    missing_act = [c for c in required_act_cols if c not in act.columns]
    if missing_act:
        raise KeyError(f"Missing columns in act section data: {missing_act}")

    # 2) Time + country + shrimp filtering.
    refusals["REFUSAL_DATE"] = pd.to_datetime(refusals["REFUSAL_DATE"], errors="coerce")
    refusals = refusals[(refusals["REFUSAL_DATE"] >= START_DATE) & (refusals["REFUSAL_DATE"] <= END_DATE)]
    refusals = refusals[refusals["ISO_CNTRY_CODE"].isin(TARGET_COUNTRIES)]

    product_col = "PRDCT_CODE_DESC_TEXT" if "PRDCT_CODE_DESC_TEXT" in refusals.columns else None
    if product_col is None:
        raise KeyError("Column PRDCT_CODE_DESC_TEXT is required to identify shrimp/prawn products.")

    shrimp_mask = (
        refusals[product_col]
        .astype(str)
        .str.upper()
        .str.contains(r"\bSHRIMP\b|\bPRAWN\b", regex=True, na=False)
    )
    refusals = refusals[shrimp_mask].copy()
    refusals["ORDER_ID"] = build_order_id(refusals)

    # 3) Expand REFUSAL_CHARGES (many ASC_IDs per refusal line).
    exploded = refusals[["ORDER_ID", "ISO_CNTRY_CODE", "REFUSAL_DATE", "REFUSAL_CHARGES"]].copy()
    exploded["ASC_ID"] = exploded["REFUSAL_CHARGES"].apply(extract_asc_ids)
    exploded = exploded.explode("ASC_ID", ignore_index=True)
    exploded = exploded[exploded["ASC_ID"].notna()].copy()
    exploded["ASC_ID"] = exploded["ASC_ID"].astype(str)

    act = act.copy()
    act["ASC_ID"] = act["ASC_ID"].astype(str).str.extract(r"(\d+)", expand=False)
    act = act[act["ASC_ID"].notna()].copy()
    act["REASON_CLASS"] = act.apply(
        lambda r: classify_reason(r.get("SCTN_NAME", ""), r.get("CHRG_STMNT_TEXT", "")), axis=1
    )

    merged = exploded.merge(act[["ASC_ID", "REASON_CLASS", "CHRG_STMNT_TEXT"]], on="ASC_ID", how="left")
    merged["REASON_CLASS"] = merged["REASON_CLASS"].fillna("OTHER")
    merged["CHRG_STMNT_TEXT"] = merged["CHRG_STMNT_TEXT"].fillna("")

    # Build extracted detailed output with requested attributes.
    detailed = merged[["ORDER_ID", "ISO_CNTRY_CODE", "REFUSAL_DATE", "REASON_CLASS", "CHRG_STMNT_TEXT", "REFUSAL_CHARGES"]].copy()
    detailed["Importing Market"] = "United States"
    detailed["Exporting country"] = detailed["ISO_CNTRY_CODE"].map(COUNTRY_NAMES).fillna(detailed["ISO_CNTRY_CODE"])
    detailed["Refusal date - formating as MM-YYYY"] = detailed["REFUSAL_DATE"].dt.strftime("%m-%Y")
    detailed["Specific Reason of Rejection"] = detailed["CHRG_STMNT_TEXT"].where(
        detailed["CHRG_STMNT_TEXT"].astype(str).str.strip().ne(""),
        detailed["REFUSAL_CHARGES"].astype(str),
    )
    detailed["Barrier Classification"] = detailed["REASON_CLASS"]
    detailed["Unique Order ID"] = detailed["ORDER_ID"]

    # If one line is BOTH, split into two records so TBT and SPS are kept separate.
    both_rows = detailed[detailed["Barrier Classification"] == "BOTH"].copy()
    if not both_rows.empty:
        tbt_rows = both_rows.copy()
        tbt_rows["Barrier Classification"] = "TBT"
        sps_rows = both_rows.copy()
        sps_rows["Barrier Classification"] = "SPS"
        detailed = pd.concat(
            [detailed[detailed["Barrier Classification"] != "BOTH"], tbt_rows, sps_rows],
            ignore_index=True,
        )

    detailed = detailed[detailed["Barrier Classification"].isin(["TBT", "SPS"])].copy()
    detailed = detailed.drop_duplicates(
        ["Unique Order ID", "Barrier Classification", "Specific Reason of Rejection"]
    )
    detailed = detailed.sort_values(
        ["Exporting country", "Refusal date - formating as MM-YYYY", "Unique Order ID", "Barrier Classification"]
    )

    detailed_output = detailed[
        [
            "Importing Market",
            "Exporting country",
            "Refusal date - formating as MM-YYYY",
            "Barrier Classification",
            "Specific Reason of Rejection",
            "Unique Order ID",
        ]
    ].copy()

    print("\nExtracted dataframe (requested attributes):")
    print(detailed_output.to_string(index=False))

    detail_file = base_dir / "FDA_extracted_shrimp_rejections_2019_Feb2026.csv"
    detailed_output.to_csv(detail_file, index=False)
    print(f"\nSaved extracted dataframe to: {detail_file}")

    # 4) Build separate counts: one order can be counted in TBT and/or SPS.
    order_reason = detailed[["Unique Order ID", "ISO_CNTRY_CODE", "Barrier Classification"]].drop_duplicates()
    tbt_orders = order_reason[order_reason["Barrier Classification"] == "TBT"].drop_duplicates(
        ["Unique Order ID", "ISO_CNTRY_CODE"]
    )
    sps_orders = order_reason[order_reason["Barrier Classification"] == "SPS"].drop_duplicates(
        ["Unique Order ID", "ISO_CNTRY_CODE"]
    )

    tbt_count = tbt_orders.groupby("ISO_CNTRY_CODE").size().rename("TBT_rejected_orders")
    sps_count = sps_orders.groupby("ISO_CNTRY_CODE").size().rename("SPS_rejected_orders")

    summary = pd.DataFrame(index=sorted(TARGET_COUNTRIES))
    summary = summary.join(tbt_count, how="left").join(sps_count, how="left").fillna(0).astype(int)
    summary.insert(0, "Country", [TARGET_COUNTRIES[c] for c in summary.index])

    print("\nRejected shrimp export orders (US FDA), 2019-01-01 to 2026-02-28")
    print(summary)

    out_file = base_dir / "shrimp_refusal_TBT_SPS_summary_2019_Feb2026.csv"
    summary.reset_index(names="ISO_CNTRY_CODE").to_csv(out_file, index=False)
    print(f"\nSaved summary to: {out_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)