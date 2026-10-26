import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path


YEAR = 2024
EXCLUDED_CATEGORIES = {"Margarine", "Butter with vegetable oil"}
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
CALENDAR_FIELDS = (
    "iso_weekday",
    "is_weekend",
    "is_sunday",
    "is_named_holiday",
    "is_general_holiday",
    "holiday_name_en",
)
OUTPUT_FIELDS = (
    "date",
    *CALENDAR_FIELDS,
    *WEATHER_FIELDS,
    "price_data_available",
    "dairy_sku_count",
    "price_observation_count",
    "promo_sku_count",
    "price_change_count",
    "promotional_offers_json",
    "price_changes_json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


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


def json_cell(items: list[dict]) -> str:
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build one daily 2024 calendar/weather row with dairy offers "
            "and observed price changes."
        )
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=next(script_dir.glob("01_*.csv"), None),
        help="Daily dairy price CSV (default: the matching 01_*.csv beside this script).",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=next(script_dir.glob("02_*.csv"), None),
        help="Holiday calendar CSV (default: the matching 02_*.csv beside this script).",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=next(script_dir.glob("03_*.csv"), None),
        help="Daily weather CSV (default: the matching 03_*.csv beside this script).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=script_dir / "daily_features_2024.csv",
        help="Output CSV path (default: daily_features_2024.csv beside this script).",
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
        date(YEAR, 1, 1).fromordinal(ordinal).isoformat()
        for ordinal in range(
            date(YEAR, 1, 1).toordinal(),
            date(YEAR + 1, 1, 1).toordinal(),
        )
    }
    if set(calendar) != expected_dates:
        missing = sorted(expected_dates - set(calendar))
        extra = sorted(set(calendar) - expected_dates)
        raise ValueError(
            f"Calendar must cover all {YEAR} dates exactly once; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    if set(weather) != expected_dates:
        missing = sorted(expected_dates - set(weather))
        extra = sorted(set(weather) - expected_dates)
        raise ValueError(
            f"Weather must cover all {YEAR} dates exactly once; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )

    prices_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    previous_price_by_sku: dict[str, tuple[str, Decimal]] = {}
    changes_by_date: dict[str, list[dict]] = defaultdict(list)
    seen_sku_dates = set()
    for row in read_csv(args.prices):
        row_date = row.get("archive_date", "")
        if not row_date.startswith(f"{YEAR}-"):
            continue
        if row.get("Category 3", "").strip() in EXCLUDED_CATEGORIES:
            continue

        sku = row.get("Product Code", "").strip()
        if not sku:
            raise ValueError(f"Price row on {row_date} has no Product Code.")
        if (row_date, sku) in seen_sku_dates:
            raise ValueError(f"Duplicate price observation for {sku} on {row_date}.")
        seen_sku_dates.add((row_date, sku))
        price = Decimal(row["Price"])
        if price <= 0:
            raise ValueError(f"Non-positive price for {sku} on {row_date}.")
        prices_by_date[row_date].append(row)

        previous = previous_price_by_sku.get(sku)
        if previous is not None and previous[1] != price:
            previous_date, previous_price = previous
            changes_by_date[row_date].append(
                {
                    "product_code": sku,
                    "product_name": row.get("Product Name", ""),
                    "brand": row.get("Brand", ""),
                    "previous_price_date": previous_date,
                    "previous_price": decimal_text(previous_price),
                    "price": decimal_text(price),
                    "change": decimal_text(price - previous_price),
                }
            )
        previous_price_by_sku[sku] = (row_date, price)

    output_rows = []
    for row_date in sorted(expected_dates):
        day_prices = prices_by_date.get(row_date, [])
        offers = [
            {
                "product_code": row["Product Code"],
                "product_name": row.get("Product Name", ""),
                "brand": row.get("Brand", ""),
                "category": row.get("Category 3", ""),
                "price": row.get("Price", ""),
                "offer": row.get("Offers", "").strip(),
            }
            for row in day_prices
            if row.get("Offers", "").strip()
        ]
        price_changes = changes_by_date.get(row_date, [])
        row = {"date": row_date}
        row.update({field: calendar[row_date].get(field, "") for field in CALENDAR_FIELDS})
        row.update({field: weather[row_date].get(field, "") for field in WEATHER_FIELDS})
        row.update(
            {
                "price_data_available": str(bool(day_prices)).lower(),
                "dairy_sku_count": len({item["Product Code"] for item in day_prices}),
                "price_observation_count": len(day_prices),
                "promo_sku_count": len(offers),
                "price_change_count": len(price_changes),
                "promotional_offers_json": json_cell(offers),
                "price_changes_json": json_cell(price_changes),
            }
        )
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    unavailable_dates = [
        row["date"] for row in output_rows if row["price_data_available"] == "false"
    ]
    print(
        f"Saved {len(output_rows)} daily rows for {YEAR} to {args.output}; "
        f"{sum(row['promo_sku_count'] for row in output_rows)} promo SKU-days; "
        f"{sum(row['price_change_count'] for row in output_rows)} observed price changes."
    )
    if unavailable_dates:
        print(
            "Price data unavailable (not treated as zero offers/changes): "
            + ", ".join(unavailable_dates)
        )


if __name__ == "__main__":
    main()
