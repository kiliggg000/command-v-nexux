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


def store_rows(input_path: Path) -> tuple[list[dict[str, str]], int]:
    region = region_name(input_path)
    pages = load_pages(input_path)
    warn_about_page_coverage(input_path, pages)
    rows = []
    for page in pages:
        for store in page["stores"]:
            address = store.get("address") or {}
            rows.append(
                {
                    "Region": region,
                    "Store Name": store.get("displayName")
                    or store.get("description")
                    or store.get("name", ""),
                    "Address": clean_text(
                        address.get("formattedAddress")
                        or address.get("displayFullAddress")
                        or address.get("addressLine1", "")
                    ),
                    "Opening Hours": opening_hours(store),
                    "Phone": address.get("phone") or address.get("phone1") or "",
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
    args = parser.parse_args()

    input_files = args.files or sorted(Path(__file__).parent.glob("stores_*.json"))
    if not input_files:
        parser.error("No input JSON files found. Pass one or more JSON paths.")

    rows_by_id = {}
    page_count = 0
    for input_path in input_files:
        file_rows, file_page_count = store_rows(input_path)
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
            fieldnames=["Region", "Store Name", "Address", "Opening Hours", "Phone"],
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
