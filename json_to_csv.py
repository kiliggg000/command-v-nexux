import argparse
import csv
import json
import re
import sys
from pathlib import Path


def clean_text(value: str) -> str:
    value = " ".join(value.split())
    return re.sub(r"\s*,\s*,+\s*", ", ", value).strip(" ,")


def region_name(input_path: Path) -> str:
    name = input_path.stem.removeprefix("stores_")
    return re.sub(r"(?i)(?:[_-](?:page|p)[_-]?\d+|[_-]\d+)$", "", name)


def opening_hours(store: dict) -> str:
    schedule = store.get("openingHours") or {}
    periods = []
    for entry in schedule.get("weekDayOpeningList") or []:
        note = entry.get("note")
        if note:
            period = note.rsplit("^", 1)[-1]
            period = re.sub(
                r"^(\d{2})(\d{2})-(\d{2})(\d{2})$",
                r"\1:\2-\3:\4",
                period,
            )
            periods.append(period)
    return "; ".join(periods)


def load_pages(input_path: Path) -> list[dict]:
    text = input_path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    pages = []
    position = 0

    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break

        page, position = decoder.raw_decode(text, position)
        if not isinstance(page, dict) or not isinstance(page.get("stores"), list):
            raise ValueError(
                f"Each JSON page in {input_path} must contain a 'stores' list."
            )
        pages.append(page)

    if not pages:
        raise ValueError(f"No JSON pages found in {input_path}.")
    return pages


def ring_location(ring: list[list[float]], longitude: float, latitude: float) -> int:
    inside = False
    for index, (x1, y1, *_) in enumerate(ring):
        x2, y2, *_ = ring[(index + 1) % len(ring)]
        cross = (longitude - x1) * (y2 - y1) - (latitude - y1) * (x2 - x1)
        if (
            abs(cross) < 1e-10
            and min(x1, x2) - 1e-10 <= longitude <= max(x1, x2) + 1e-10
            and min(y1, y2) - 1e-10 <= latitude <= max(y1, y2) + 1e-10
        ):
            return 0

        if (y1 > latitude) != (y2 > latitude):
            crossing_longitude = x1 + (latitude - y1) * (x2 - x1) / (y2 - y1)
            if longitude < crossing_longitude:
                inside = not inside
    return 1 if inside else -1


def polygon_covers(
    coordinates: list[list[list[float]]], longitude: float, latitude: float
) -> bool:
    outer_ring, *holes = coordinates
    if ring_location(outer_ring, longitude, latitude) == -1:
        return False
    return all(ring_location(hole, longitude, latitude) != 1 for hole in holes)


def geometry_covers(geometry: dict, longitude: float, latitude: float) -> bool:
    if geometry.get("type") == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry.get("type") == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"Unsupported district geometry: {geometry.get('type')!r}")
    return any(
        polygon_covers(polygon, longitude, latitude) for polygon in polygons
    )


def load_districts(boundary_path: Path) -> list[dict]:
    with boundary_path.open(encoding="utf-8-sig", newline="") as source:
        districts = []
        for row in csv.DictReader(source):
            districts.append(
                {
                    "code": row["had_district_code"],
                    "english": row["district_en"],
                    "chinese": row["district_zh"],
                    "geometry": json.loads(row["geometry_geojson"]),
                }
            )
    if not districts:
        raise ValueError(f"No district boundaries found in {boundary_path}.")
    return districts


def find_district(
    store: dict, districts: list[dict], input_path: Path
) -> tuple[str, str, str]:
    point = store.get("geoPoint") or {}
    latitude = point.get("latitude")
    longitude = point.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError(
            f"Store {store.get('displayName') or store.get('name')!r} in "
            f"{input_path} has no geoPoint coordinates."
        )

    matches = [
        district
        for district in districts
        if geometry_covers(district["geometry"], longitude, latitude)
    ]
    if len(matches) != 1:
        store_name = store.get("displayName") or store.get("name")
        if not matches:
            raise ValueError(
                f"Could not match store {store_name!r} at "
                f"({latitude}, {longitude}) to a district boundary."
            )
        raise ValueError(
            f"Store {store_name!r} at ({latitude}, {longitude}) matches "
            f"multiple district boundaries: "
            f"{', '.join(district['english'] for district in matches)}."
        )

    district = matches[0]
    return district["code"], district["english"], district["chinese"]


def warn_about_page_coverage(input_path: Path, pages: list[dict]) -> None:
    pagination = [page.get("pagination") or {} for page in pages]
    current_pages = [
        page.get("currentPage")
        for page in pagination
        if isinstance(page.get("currentPage"), int)
    ]
    total_pages = [
        page.get("totalPages")
        for page in pagination
        if isinstance(page.get("totalPages"), int)
    ]
    if not current_pages or not total_pages:
        return

    duplicates = sorted(
        page_number
        for page_number in set(current_pages)
        if current_pages.count(page_number) > 1
    )
    missing = sorted(set(range(max(total_pages))) - set(current_pages))
    if duplicates:
        print(
            f"Warning: {input_path} contains duplicate page numbers: "
            f"{', '.join(map(str, duplicates))}.",
            file=sys.stderr,
        )
    if missing:
        print(
            f"Warning: {input_path} is missing pages: "
            f"{', '.join(map(str, missing))}.",
            file=sys.stderr,
        )


def store_rows(
    input_path: Path, districts: list[dict]
) -> tuple[list[dict[str, str]], int]:
    region = region_name(input_path)
    pages = load_pages(input_path)
    warn_about_page_coverage(input_path, pages)
    rows = []
    for page in pages:
        for store in page["stores"]:
            address = store.get("address") or {}
            district_code, district_english, district_chinese = find_district(
                store, districts, input_path
            )
            point = store["geoPoint"]
            rows.append(
                {
                    "Region": region,
                    "Store Type": store.get("storeContent", ""),
                    "District Code": district_code,
                    "District English": district_english,
                    "District Chinese": district_chinese,
                    "Store Name": store.get("displayName")
                    or store.get("description")
                    or store.get("name", ""),
                    "Address": clean_text(
                        address.get("formattedAddress")
                        or address.get("displayFullAddress")
                        or address.get("addressLine1", "")
                    ),
                    "Opening Hours": opening_hours(store),
                    "Fast Delivery": store.get("elabFastDelivery", ""),
                    "Click & Collect": any(
                        feature.get("name") == "Click & Collect"
                        for feature in store.get("elabFeatures") or []
                    ),
                    "Pay & Collect": store.get("elabPayCollect", ""),
                    "Pickup Express Eligible": store.get(
                        "pickupExpressEligible", ""
                    ),
                    "Latitude": point["latitude"],
                    "Longitude": point["longitude"],
                    "_store_id": store.get("name") or address.get("id"),
                }
            )
    return rows, len(pages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PNS store JSON files into one CSV."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="JSON files to convert (default: stores_*.json beside this script).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pns_stores.csv"),
        help="CSV output path (default: pns_stores.csv).",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        help="District boundary CSV (default: the single 07_*.csv beside this script).",
    )
    args = parser.parse_args()

    input_files = args.files or sorted(Path(__file__).parent.glob("stores_*.json"))
    if not input_files:
        parser.error("No input JSON files found. Pass one or more JSON paths.")

    if args.boundaries:
        boundary_path = args.boundaries
    else:
        boundary_files = sorted(Path(__file__).parent.glob("07_*.csv"))
        if len(boundary_files) != 1:
            parser.error(
                "Expected exactly one 07_*.csv beside this script; "
                "pass the correct file with --boundaries."
            )
        boundary_path = boundary_files[0]
    districts = load_districts(boundary_path)

    rows_by_id = {}
    page_count = 0
    for input_path in input_files:
        file_rows, file_page_count = store_rows(input_path, districts)
        page_count += file_page_count
        for row in file_rows:
            store_id = row.pop("_store_id") or row["Store Name"]
            rows_by_id[(row["Region"], store_id)] = row
    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (row["Region"].casefold(), row["Store Name"].casefold()),
    )

    with args.output.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "Region",
                "Store Type",
                "District Code",
                "District English",
                "District Chinese",
                "Store Name",
                "Address",
                "Opening Hours",
                "Fast Delivery",
                "Click & Collect",
                "Pay & Collect",
                "Pickup Express Eligible",
                "Latitude",
                "Longitude",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {len(rows)} unique stores from {page_count} pages in "
        f"{len(input_files)} files "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
