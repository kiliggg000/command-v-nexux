import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path


YEAR = 2024
EXCLUDED_CATEGORIES = {"Margarine", "Butter with vegetable oil"}
CALENDAR_FIELDS = (
    "iso_weekday",
    "is_weekend",
    "is_sunday",
    "is_named_holiday",
    "is_general_holiday",
    "holiday_name_en",
)
WEATHER_FIELDS = (
    "mean_pressure_hpa",
    "max_temperature_c",
    "mean_temperature_c",
    "min_temperature_c",
    "mean_dewpoint_c",
    "mean_relative_humidity_pct",
    "mean_cloud_pct",
    "rainfall_mm",
    "rainfall_raw",
    "rainfall_trace",
    "sunshine_hours",
    "prevailing_wind_direction_deg",
    "mean_wind_speed_kmh",
)
PANEL_FIELDS = (
    "date",
    "Product Code",
    "Product Name",
    "Brand",
    "Category 2",
    "Category 3",
    "Price",
    "Offers",
    "promo_flag",
    "promo_type",
    "promo_parse_status",
    "buy_qty",
    "save_amount",
    "discount_pct",
    "depth_pct",
    "effective_unit_price",
    "previous_observed_price_date",
    "previous_observed_price",
    "price_change",
    "price_changed",
    *CALENDAR_FIELDS,
    *WEATHER_FIELDS,
)
DAILY_FIELDS = (
    "date",
    *CALENDAR_FIELDS,
    *WEATHER_FIELDS,
    "price_data_available",
    "dairy_sku_count",
    "price_observation_count",
    "promo_sku_count",
    "promo_parseable_sku_count",
    "promo_parse_rate",
    "mean_promo_depth_pct",
    "price_change_count",
)
MONEY = r"\$\s*(\d+(?:\.\d{1,2})?)"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def index_by_date(
    rows: list[dict[str, str]], source_name: str
) -> dict[str, dict[str, str]]:
    indexed = {}
    for row in rows:
        row_date = row.get("date", "")
        if row_date in indexed:
            raise ValueError(f"{source_name} has duplicate records for {row_date}.")
        indexed[row_date] = row
    return indexed


def parse_offer(offer: str, regular_price: Decimal) -> dict:
    parsed = {
        "promo_flag": bool(offer.strip()),
        "promo_type": "none",
        "promo_parse_status": "no_offer",
        "buy_qty": None,
        "save_amount": None,
        "discount_pct": None,
        "depth_pct": None,
        "effective_unit_price": None,
    }
    if not offer.strip():
        return parsed

    text = re.sub(r"\s+", " ", offer).strip()
    candidates = []
    patterns = (
        (
            "multi_buy_saving",
            rf"\bbuy\s+(\d+)\s+(?:to\s+)?save\s+{MONEY}",
        ),
        (
            "multi_buy_saving",
            rf"\bpurchase\s+(\d+)\s+(?:items?|pcs?)\b.*?\bsave\s+{MONEY}",
        ),
        (
            "buy_get_free",
            r"\bbuy\s+(\d+)\s+get\s+(\d+)\s+free\b",
        ),
        (
            "multi_buy_price",
            rf"\bbuy\s+(\d+)\s+(?:at|for)\s+{MONEY}",
        ),
        (
            "second_item_discount",
            rf"\bbuy\s+second\s+for\s+{MONEY}",
        ),
        (
            "percentage_discount",
            r"\b(?:enjoy\s+)?(\d+(?:\.\d+)?)\s*%\s*off\b",
        ),
    )

    for promo_type, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if promo_type == "multi_buy_saving":
                quantity, saving = int(groups[0]), Decimal(groups[1])
                total_regular = regular_price * quantity
                if quantity > 0 and Decimal(0) <= saving < total_regular:
                    effective = (total_regular - saving) / quantity
                    candidates.append(
                        (saving / total_regular, promo_type, quantity, saving, effective)
                    )
            elif promo_type == "buy_get_free":
                paid_quantity, free_quantity = map(int, groups)
                total_quantity = paid_quantity + free_quantity
                if paid_quantity > 0 and free_quantity > 0:
                    saving = regular_price * free_quantity
                    effective = regular_price * paid_quantity / total_quantity
                    candidates.append(
                        (
                            saving / (regular_price * total_quantity),
                            promo_type,
                            total_quantity,
                            saving,
                            effective,
                        )
                    )
            elif promo_type == "multi_buy_price":
                quantity, bundle_price = int(groups[0]), Decimal(groups[1])
                total_regular = regular_price * quantity
                if quantity > 0 and Decimal(0) < bundle_price < total_regular:
                    saving = total_regular - bundle_price
                    candidates.append(
                        (
                            saving / total_regular,
                            promo_type,
                            quantity,
                            saving,
                            bundle_price / quantity,
                        )
                    )
            elif promo_type == "second_item_discount":
                second_price = Decimal(groups[0])
                if Decimal(0) <= second_price < regular_price:
                    saving = regular_price - second_price
                    candidates.append(
                        (
                            saving / (regular_price * 2),
                            promo_type,
                            2,
                            saving,
                            (regular_price + second_price) / 2,
                        )
                    )
            else:
                discount_pct = Decimal(groups[0])
                if Decimal(0) < discount_pct < Decimal(100):
                    quantity_match = re.search(
                        r"\b(?:buy|purchase)\s+(\d+)\s+(?:items?|pcs?)\b",
                        text,
                        re.IGNORECASE,
                    )
                    quantity = int(quantity_match.group(1)) if quantity_match else None
                    saving = (
                        regular_price * (quantity or 1) * discount_pct / Decimal(100)
                    )
                    candidates.append(
                        (
                            discount_pct / Decimal(100),
                            promo_type,
                            quantity,
                            saving,
                            regular_price * (Decimal(1) - discount_pct / Decimal(100)),
                        )
                    )

    if not candidates:
        parsed["promo_type"] = "other"
        parsed["promo_parse_status"] = "unparsed"
        return parsed

    depth, promo_type, quantity, saving, effective = max(
        candidates, key=lambda item: item[0]
    )
    if len(candidates) > 1 or "/" in text:
        promo_type = "compound"

    parsed.update(
        {
            "promo_type": promo_type,
            "promo_parse_status": "parsed",
            "buy_qty": quantity,
            "save_amount": saving,
            "discount_pct": depth * Decimal(100),
            "depth_pct": depth,
            "effective_unit_price": effective,
        }
    )
    if len(candidates) > 1:
        parsed["promo_parse_status"] = "parsed_compound_primary_offer"
    return parsed


def decimal_cell(value: Decimal | None) -> str:
    return "" if value is None else format(value, "f")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a flat 2024 SKU-day price panel and daily context table."
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=next(script_dir.glob("01_*.csv"), None),
        help="Daily dairy price CSV (default: matching 01_*.csv beside this script).",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=next(script_dir.glob("02_*.csv"), None),
        help="Holiday calendar CSV (default: matching 02_*.csv beside this script).",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=next(script_dir.glob("03_*.csv"), None),
        help="Daily weather CSV (default: matching 03_*.csv beside this script).",
    )
    parser.add_argument(
        "--panel-output",
        type=Path,
        default=script_dir / "sku_daily_prices_2024.csv",
        help="Flat SKU-day output path.",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=script_dir / "daily_features_2024.csv",
        help="One-row-per-date context output path.",
    )
    args = parser.parse_args()
    if not args.prices or not args.calendar or not args.weather:
        parser.error("Could not find one or more required 01_, 02_, or 03_ CSV files.")

    calendar = {
        key: row
        for key, row in index_by_date(read_csv(args.calendar), "Calendar").items()
        if key.startswith(f"{YEAR}-")
    }
    weather = {
        key: row
        for key, row in index_by_date(read_csv(args.weather), "Weather").items()
        if key.startswith(f"{YEAR}-")
    }
    expected_dates = {
        date.fromordinal(ordinal).isoformat()
        for ordinal in range(
            date(YEAR, 1, 1).toordinal(),
            date(YEAR + 1, 1, 1).toordinal(),
        )
    }
    for label, indexed in (("Calendar", calendar), ("Weather", weather)):
        if set(indexed) != expected_dates:
            missing = sorted(expected_dates - set(indexed))
            extra = sorted(set(indexed) - expected_dates)
            raise ValueError(
                f"{label} must cover all {YEAR} dates exactly once; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )

    source_rows = [
        row
        for row in read_csv(args.prices)
        if row.get("archive_date", "").startswith(f"{YEAR}-")
        and row.get("Category 3", "").strip() not in EXCLUDED_CATEGORIES
    ]
    source_rows.sort(key=lambda row: (row["archive_date"], row["Product Code"]))

    panel_rows = []
    daily_prices: dict[str, list[dict]] = defaultdict(list)
    previous_price_by_sku: dict[str, tuple[str, Decimal]] = {}
    seen_sku_dates = set()
    promo_counts = Counter()
    promo_parse_counts = Counter()
    price_change_counts = Counter()
    promo_depths: dict[str, list[Decimal]] = defaultdict(list)

    for source in source_rows:
        row_date = source["archive_date"]
        sku = source.get("Product Code", "").strip()
        if not sku:
            raise ValueError(f"Price row on {row_date} has no Product Code.")
        if (row_date, sku) in seen_sku_dates:
            raise ValueError(f"Duplicate price observation for {sku} on {row_date}.")
        seen_sku_dates.add((row_date, sku))

        price = Decimal(source["Price"])
        if price <= 0:
            raise ValueError(f"Non-positive price for {sku} on {row_date}.")
        promo = parse_offer(source.get("Offers", ""), price)
        previous = previous_price_by_sku.get(sku)
        previous_date = previous[0] if previous else ""
        previous_price = previous[1] if previous else None
        change = price - previous_price if previous_price is not None else None
        previous_price_by_sku[sku] = (row_date, price)

        row = {
            "date": row_date,
            "Product Code": sku,
            "Product Name": source.get("Product Name", ""),
            "Brand": source.get("Brand", ""),
            "Category 2": source.get("Category 2", ""),
            "Category 3": source.get("Category 3", ""),
            "Price": format(price, "f"),
            "Offers": source.get("Offers", "").strip(),
            **{
                key: (
                    decimal_cell(value)
                    if isinstance(value, Decimal)
                    else str(value).lower()
                    if isinstance(value, bool)
                    else value if value is not None else ""
                )
                for key, value in promo.items()
            },
            "previous_observed_price_date": previous_date,
            "previous_observed_price": decimal_cell(previous_price),
            "price_change": decimal_cell(change),
            "price_changed": str(change is not None and change != 0).lower(),
        }
        row.update({key: calendar[row_date].get(key, "") for key in CALENDAR_FIELDS})
        row.update({key: weather[row_date].get(key, "") for key in WEATHER_FIELDS})
        panel_rows.append(row)
        daily_prices[row_date].append(row)

        if promo["promo_flag"]:
            promo_counts[row_date] += 1
            if promo["promo_parse_status"].startswith("parsed"):
                promo_parse_counts[row_date] += 1
                promo_depths[row_date].append(promo["depth_pct"])
        if change is not None and change != 0:
            price_change_counts[row_date] += 1

    daily_rows = []
    for row_date in sorted(expected_dates):
        day_rows = daily_prices.get(row_date, [])
        parsed_count = promo_parse_counts[row_date]
        offer_count = promo_counts[row_date]
        depths = promo_depths[row_date]
        daily_row = {"date": row_date}
        daily_row.update({key: calendar[row_date].get(key, "") for key in CALENDAR_FIELDS})
        daily_row.update({key: weather[row_date].get(key, "") for key in WEATHER_FIELDS})
        daily_row.update(
            {
                "price_data_available": str(bool(day_rows)).lower(),
                "dairy_sku_count": len({row["Product Code"] for row in day_rows}),
                "price_observation_count": len(day_rows),
                "promo_sku_count": offer_count,
                "promo_parseable_sku_count": parsed_count,
                "promo_parse_rate": (
                    format(Decimal(parsed_count) / Decimal(offer_count), ".6f")
                    if offer_count
                    else ""
                ),
                "mean_promo_depth_pct": (
                    format(
                        sum(depths, Decimal(0)) / Decimal(len(depths)),
                        ".6f",
                    )
                    if depths
                    else ""
                ),
                "price_change_count": price_change_counts[row_date],
            }
        )
        daily_rows.append(daily_row)

    write_csv(args.panel_output, PANEL_FIELDS, panel_rows)
    write_csv(args.daily_output, DAILY_FIELDS, daily_rows)

    promo_rows = [row for row in panel_rows if row["promo_flag"] == "true"]
    parsed_rows = [
        row for row in promo_rows if row["promo_parse_status"].startswith("parsed")
    ]
    parse_rate = len(parsed_rows) / len(promo_rows) if promo_rows else 1.0
    print(
        f"Saved {len(panel_rows)} flat SKU-day rows to {args.panel_output}; "
        f"{len(daily_rows)} daily rows to {args.daily_output}."
    )
    print(
        f"Parsed {len(parsed_rows)}/{len(promo_rows)} promotional SKU-days "
        f"({parse_rate:.2%}); "
        f"{sum(price_change_counts.values())} observed price changes."
    )
    unparsed = Counter(
        row["Offers"]
        for row in promo_rows
        if not row["promo_parse_status"].startswith("parsed")
    )
    if unparsed:
        print("Unparsed offer examples:")
        for offer, count in unparsed.most_common(10):
            print(f"  {count}: {offer}")
    if parse_rate < 0.95:
        raise SystemExit("Promotion parse rate is below the required 95% threshold.")


if __name__ == "__main__":
    main()
