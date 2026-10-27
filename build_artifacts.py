#!/usr/bin/env python
"""Build the three generation-gating artifacts:
  B3/B9  sku_master.csv      (sku -> type, brand, category, pack_kg)
  B6     store_weights.csv   (store -> district population weight)
Verified inputs: sku_daily_prices_2024.csv (120 dairy SKUs), pns_stores.csv (228),
04_...2021 census (18 districts, population at column index 4).
"""
import csv, re, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY_LIQUID = 1.03  # kg per litre for milk/cream

VOL = re.compile(r"(\d+(?:\.\d+)?)\s*(mL|ml|ML|L|litre|g|kg|KG)\b")
MULT = re.compile(r"[xX\u00d7]\s*(\d+)\b")

def pack_kg(name):
    m = VOL.search(name)
    if not m:
        return None
    q = float(m.group(1)); u = m.group(2).lower()
    if u == "l" or u == "litre":
        kg = q * DENSITY_LIQUID
    elif u == "ml":
        kg = q / 1000.0 * DENSITY_LIQUID
    elif u == "kg":
        kg = q
    else:  # g
        kg = q / 1000.0
    k = MULT.search(name)
    if k:
        kg *= int(k.group(1))
    return round(kg, 4)

def map_type(cat3):
    c = cat3.strip()
    if c.startswith("UHT"): return "UHT milk"
    if c.startswith("Chilled"): return "Chilled fresh milk"
    if c.startswith("Cheese slices") or "Hard cheese" in c: return "Cheese"
    if c == "Yoghurt": return "Yoghurt"
    if c == "Butter": return "Butter"
    if c.startswith("Evaporated"): return "Evaporated milk"
    if c.startswith("Sweetened"): return "Condensed milk"
    if c.startswith("Cheese spreads"): return "Cheese spread"
    if c.startswith("Yoghurt drinks") or c.startswith("Yoghurt flavour"): return "Yoghurt drinks"
    if c.startswith("Whipping"): return "Cream"
    return "UNMAPPED"

# ---- SKU master (B3 + B9) ----
src = os.path.join(HERE, "sku_daily_prices_2024.csv")
skus = {}
with open(src, encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        if r["Product Code"] in skus:
            continue
        skus[r["Product Code"]] = (
            r["Product Name"], r["Brand"], r["Category 2"], r["Category 3"])

rows_out = []
unparsed = []
for code, (name, brand, c2, c3) in skus.items():
    pk = pack_kg(name)
    if pk is None:
        unparsed.append((code, name))
    rows_out.append({
        "Product Code": code, "Product Name": name, "Brand": brand,
        "Category 2": c2, "Category 3": c3, "type": map_type(c3),
        "pack_kg": pk if pk is not None else ""})

types = collections.Counter(r["type"] for r in rows_out)
print("SKU master: %d rows" % len(rows_out))
print("types:", dict(types))
print("unparsed pack size:", len(unparsed), unparsed[:5])
assert types["UNMAPPED"] == 0, "unmapped types present"
assert len(unparsed) == 0, "pack size unparsed"

with open(os.path.join(HERE, "sku_master.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["Product Code", "Product Name", "Brand",
        "Category 2", "Category 3", "type", "pack_kg"])
    w.writeheader(); w.writerows(rows_out)
print("wrote sku_master.csv")

# ---- Store weights (B6) ----
# Recipe: uniform weight across stores within a district, scaled by district
# population; normalised to sum 1. Documented in Data_Plan B6.
stores = list(csv.DictReader(open(os.path.join(HERE, "pns_stores.csv"),
    encoding="utf-8-sig", newline="")))
census = list(csv.DictReader(open(os.path.join(HERE,
    "04_香港18区人口年龄收入及住户资料_2021普查_中文列标题.csv"),
    encoding="utf-8-sig", newline="")))

# census keys: col0 letter code, col2 English name, col4 total population
# key everything by lowercased English name for robust matching
cen = {}  # lower name -> population
for x in census:
    cols = list(x.keys())
    en = x[cols[2]].strip().lower()
    pop = int(x[cols[4]])
    cen[en] = pop

def match_district(d):
    d = d.strip().lower()
    if d in cen:
        return d
    # fallback: any census name that contains or is contained by d
    for en in cen:
        if d in en or en in d:
            return en
    return None

matched = 0; unmatched = []
store_rows = []
for s in stores:
    en = match_district(s["District English"])
    if en is None:
        unmatched.append(s["District English"]); pop = 0
    else:
        matched += 1; pop = cen[en]
    store_rows.append((s["Store Name"], s["District English"], s["District Code"],
        s["Store Type"], s["Latitude"], s["Longitude"], pop))

print("\nstore match: %d/%d matched" % (matched, len(store_rows)))
if unmatched:
    print("UNMATCHED districts:", sorted(set(unmatched)))

total_pop = sum(r[6] for r in store_rows)
by_district = collections.defaultdict(list)
for r in store_rows:
    by_district[r[1]].append(r)
out = []
for r in store_rows:
    d = r[1]
    n_in_d = len(by_district[d])
    # district share of population, split equally among its stores
    w = (r[6] / total_pop) / n_in_d if total_pop else 0
    out.append({
        "Store Name": r[0], "District English": r[1], "District Code": r[2],
        "Store Type": r[3], "Latitude": r[4], "Longitude": r[5],
        "district_pop": r[6], "store_weight": round(w, 8)})

sw = sum(x["store_weight"] for x in out)
print("store_weight sum = %.6f (target 1.0)" % sw)
for x in out:
    x["store_weight"] = round(x["store_weight"] / sw, 8)  # renormalise
print("after renorm sum = %.6f" % sum(x["store_weight"] for x in out))

with open(os.path.join(HERE, "store_weights.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["Store Name", "District English",
        "District Code", "Store Type", "Latitude", "Longitude",
        "district_pop", "store_weight"])
    w.writeheader(); w.writerows(out)
print("wrote store_weights.csv")
