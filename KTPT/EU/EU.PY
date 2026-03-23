from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd


START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2026-02-28 23:59:59")

COUNTRY_ALIASES = {
	"VN": ["VIETNAM", "VIET NAM", "VN"],
	"IN": ["INDIA", "IN"],
	"EC": ["ECUADOR", "EC"],
}

COUNTRY_NAMES = {
	"VN": "Vietnam",
	"IN": "India",
	"EC": "Ecuador",
}


def read_csv_robust(file_path: Path) -> pd.DataFrame:
	encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
	for enc in encodings:
		try:
			df = pd.read_csv(file_path, encoding=enc, low_memory=False)
			print(f"Loaded {file_path.name} with encoding={enc}")
			return df
		except UnicodeDecodeError:
			continue
		except pd.errors.ParserError:
			try:
				df = pd.read_csv(
					file_path,
					encoding=enc,
					engine="python",
					on_bad_lines="skip",
				)
				print(
					f"Loaded {file_path.name} with encoding={enc}, engine=python, on_bad_lines=skip "
					"(some malformed rows may be skipped)"
				)
				return df
			except Exception:
				continue

	# Last fallback: keep pipeline alive by replacing undecodable bytes.
	df = pd.read_csv(
		file_path,
		encoding="utf-8",
		encoding_errors="replace",
		engine="python",
		on_bad_lines="skip",
	)
	print(
		f"Loaded {file_path.name} with encoding=utf-8 (invalid bytes replaced), "
		"engine=python, on_bad_lines=skip"
	)
	return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	df.columns = [str(c).strip().lower() for c in df.columns]
	return df


def ensure_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise KeyError(f"Missing required columns: {missing}")


def map_origin_country(origin_text: str) -> str | None:
	origin_up = f" {str(origin_text).upper()} "
	for code, aliases in COUNTRY_ALIASES.items():
		for alias in aliases:
			if re.search(rf"\b{re.escape(alias)}\b", origin_up):
				return code
	return None


def is_shrimp_row(row: pd.Series) -> bool:
	text = f"{row.get('category', '')} {row.get('subject', '')}".upper()
	return bool(re.search(r"\bSHRIMP\b|\bSHRIMPS\b|\bPRAWN\b|\bPRAWNS\b", text))


def classify_reason_tbt_sps(row: pd.Series) -> str:
	text = f"{row.get('subject', '')} {row.get('hazards', '')} {row.get('classification', '')}".upper()

	tbt_keywords = [
		"LABEL",
		"LABELLING",
		"LABELING",
		"UNLABELLED",
		"MISLABEL",
		"MISLEAD",
		"DOCUMENTARY CHECK",
		"IDENTITY CHECK",
		"HEALTH CERTIFICATE",
		"CERTIFICATE",
		"PACKAGING",
		"MARKING",
		"UNAUTHORIZED NOVEL FOOD",
		"UNAUTHORISED NOVEL FOOD",
	]

	sps_keywords = [
		"SALMONELLA",
		"LISTERIA",
		"VIBRIO",
		"NOROVIRUS",
		"PATHOGENIC",
		"MICRO-ORGANISMS",
		"MYCOTOXINS",
		"AFLATOXIN",
		"OCHRATOXIN",
		"PESTICIDE",
		"RESIDUES",
		"VETERINARY DRUG",
		"ANTIBIOTIC",
		"HEAVY METALS",
		"CADMIUM",
		"MERCURY",
		"LEAD",
		"PARASITE",
		"CONTAMINATION",
		"CONTAMINANT",
		"INSANITARY",
		"SPOILAGE",
		"ROTTEN",
		"UNFAVORABLE PHYSICAL CHECK",
		"UNFAVOURABLE PHYSICAL CHECK",
	]

	has_tbt = any(k in text for k in tbt_keywords)
	has_sps = any(k in text for k in sps_keywords)

	if has_tbt and has_sps:
		return "BOTH"
	if has_tbt:
		return "TBT"
	if has_sps:
		return "SPS"
	return "OTHER"


def build_order_id(df: pd.DataFrame) -> pd.Series:
	if "reference" in df.columns:
		return df["reference"].astype(str).str.strip()
	return pd.Series([f"ROW_{i}" for i in df.index], index=df.index)


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	data_file = base_dir / "RASFF_window.csv"

	if not data_file.exists():
		raise FileNotFoundError(f"Data file not found: {data_file}")

	df = normalize_columns(read_csv_robust(data_file))
	ensure_columns(
		df,
		[
			"reference",
			"date",
			"classification",
			"origin",
			"category",
			"subject",
			"hazards",
		],
	)

	# EU rejected orders: keep border rejection notifications only.
	df["classification"] = df["classification"].astype(str)
	border_mask = df["classification"].str.contains("border rejection", case=False, na=False)
	df = df[border_mask].copy()

	# Date filter.
	df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
	df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

	# Country filter based on origin.
	df["ISO_CNTRY_CODE"] = df["origin"].apply(map_origin_country)
	df = df[df["ISO_CNTRY_CODE"].isin(COUNTRY_ALIASES)].copy()

	# Shrimp-only filter.
	df = df[df.apply(is_shrimp_row, axis=1)].copy()
	df["ORDER_ID"] = build_order_id(df)

	# Classify reason for each rejected order.
	df["REASON_CLASS"] = df.apply(classify_reason_tbt_sps, axis=1)

	# Build extracted detailed output with the exact requested attributes.
	df["Refusal date - formating as MM-YYYY"] = df["date"].dt.strftime("%m-%Y")
	df["Importing Market"] = df["notifying_country"].astype(str).replace("nan", "")
	df["Exporting country"] = df["ISO_CNTRY_CODE"].map(COUNTRY_NAMES).fillna(df["origin"].astype(str))
	df["Specific Reason of Rejection"] = df["hazards"].fillna("")
	df.loc[df["Specific Reason of Rejection"].astype(str).str.strip().eq(""), "Specific Reason of Rejection"] = (
		df["subject"].fillna("")
	)

	detailed = df[
		[
			"Importing Market",
			"Exporting country",
			"Refusal date - formating as MM-YYYY",
			"REASON_CLASS",
			"Specific Reason of Rejection",
			"ORDER_ID",
		]
	].copy()
	detailed = detailed.rename(
		columns={
			"REASON_CLASS": "Barrier Classification",
			"ORDER_ID": "Unique Order ID",
		}
	)

	# If one order is BOTH, expand into two rows so TBT and SPS stay separated.
	both_rows = detailed[detailed["Barrier Classification"] == "BOTH"].copy()
	if not both_rows.empty:
		tbt_rows = both_rows.copy()
		tbt_rows["Barrier Classification"] = "TBT"
		sps_rows = both_rows.copy()
		sps_rows["Barrier Classification"] = "SPS"
		detailed = pd.concat([detailed[detailed["Barrier Classification"] != "BOTH"], tbt_rows, sps_rows], ignore_index=True)

	# Keep only the requested barrier classes.
	detailed = detailed[detailed["Barrier Classification"].isin(["TBT", "SPS"])].copy()

	detailed = detailed.drop_duplicates().sort_values(
		["Exporting country", "Refusal date - formating as MM-YYYY", "Unique Order ID", "Barrier Classification"]
	)

	# Separate counting from detailed output.
	order_reason = detailed[["Unique Order ID", "Exporting country", "Barrier Classification"]].drop_duplicates()

	tbt_orders = order_reason[order_reason["Barrier Classification"] == "TBT"].drop_duplicates(
		["Unique Order ID", "Exporting country"]
	)
	sps_orders = order_reason[order_reason["Barrier Classification"] == "SPS"].drop_duplicates(
		["Unique Order ID", "Exporting country"]
	)

	summary = pd.DataFrame(index=["EC", "IN", "VN"])
	summary["Country"] = ["Ecuador", "India", "Vietnam"]
	summary["TBT_rejected_orders"] = (
		tbt_orders["Exporting country"].map({"Ecuador": "EC", "India": "IN", "Vietnam": "VN"}).value_counts().reindex(summary.index).fillna(0).astype(int)
	)
	summary["SPS_rejected_orders"] = (
		sps_orders["Exporting country"].map({"Ecuador": "EC", "India": "IN", "Vietnam": "VN"}).value_counts().reindex(summary.index).fillna(0).astype(int)
	)

	print("\nExtracted dataframe (requested attributes):")
	print(detailed.to_string(index=False))

	detail_file = base_dir / "EU_extracted_shrimp_rejections_2019_Feb2026.csv"
	detailed.to_csv(detail_file, index=False)
	print(f"\nSaved extracted dataframe to: {detail_file}")

	print("\nEU rejected shrimp export orders (border rejection notifications)")
	print("Period: 2019-01-01 to 2026-02-28")
	print(summary)

	output_file = base_dir / "EU_shrimp_rejected_orders_TBT_SPS_2019_Feb2026.csv"
	summary.reset_index(names="ISO_CNTRY_CODE").to_csv(output_file, index=False)
	print(f"\nSaved summary to: {output_file}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(f"ERROR: {exc}")
		sys.exit(1)
