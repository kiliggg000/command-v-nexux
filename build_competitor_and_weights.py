#!/usr/bin/env python
"""Integrate the three teammate files into the cleaning pipeline:
  - store_weights_140.csv   (B6 on the 140 flagship universe; district-pop weights)
  - competitor_features.csv (B10 realized: per-store competitor counts within 500m)
Source files (all dated 2026-10-27):
  01_百佳招牌门店地址及经纬度_140条.csv       140 flagship stores (has census tract + district)
  10_仅百佳招牌门店500米内竞争店_直接用这份.csv  competitor stores within 500m
  04_...2021 census (district population)
"""
import csv, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 140 flagship stores ----
flagship = list(csv.DictReader(open(os.path.join(HERE,
    "01_百佳招牌门店地址及经纬度_140条.csv"), encoding="utf-8-sig", newline="")))

# ---- census: match by Chinese district name ----
census = list(csv.DictReader(open(os.path.join(HERE,
    "04_香港18区人口年龄收入及住户资料_2021普查_中文列标题.csv"),
    encoding="utf-8-sig", newline="")))
cen_pop = {}
for x in census:
    cols = list(x.keys())
    zh = x[cols[3]].strip()           # 地區名稱（中文）
    cen_pop[zh] = int(x[cols[4]])      # 總人口（人）

def match_zh(d):
    d = d.strip()
    if d in cen_pop:
        return d
    for k in cen_pop:
        if d in k or k in d:
            return k
    return None

# ---- store weights on 140 (B6) ----
rows = []
unmatched = []
for s in flagship:
    zh = match_zh(s["所属行政区"])
    pop = cen_pop[zh] if zh else 0
    if zh is None:
        unmatched.append(s["所属行政区"])
    rows.append((s["本次整理门店ID"], s["门店名称"], s["所属行政区"],
        s["纬度"], s["经度"], pop))
print("140 store match:", sum(1 for r in rows if r[5] > 0), "/", len(rows),
      "| unmatched:", sorted(set(unmatched)))
total = sum(r[5] for r in rows)
by_d = collections.defaultdict(list)
for r in rows:
    by_d[r[2]].append(r)
out = []
for r in rows:
    n = len(by_d[r[2]])
    w = (r[5] / total) / n if total else 0
    out.append({"store_id": r[0], "store_name": r[1], "district_zh": r[2],
        "latitude": r[3], "longitude": r[4], "district_pop": r[5],
        "store_weight": round(w, 8)})
sw = sum(x["store_weight"] for x in out)
for x in out:
    x["store_weight"] = round(x["store_weight"] / sw, 8)
print("store_weight sum = %.6f" % sum(x["store_weight"] for x in out))
with open(os.path.join(HERE, "store_weights_140.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["store_id", "store_name", "district_zh",
        "latitude", "longitude", "district_pop", "store_weight"])
    w.writeheader(); w.writerows(out)
print("wrote store_weights_140.csv")

# ---- competitor features (B10) ----
comp = list(csv.DictReader(open(os.path.join(HERE,
    "10_仅百佳招牌门店500米内竞争店_直接用这份.csv"), encoding="utf-8-sig", newline="")))
CONV = {"7ELEVEN", "CIRCLEK", "759", "USELECT"}
SUPER = {"WELLCOME", "JASONS", "YATA", "AEON", "CITYSUPER"}
feat = collections.defaultdict(lambda: {"conv": 0, "super": 0, "other": 0,
    "nearest_all": 1e9, "nearest_conv": 1e9, "nearest_super": 1e9})
for x in comp:
    sid = x["百佳集团门店ID"]
    code = x["竞争店连锁代码"]
    d = float(x["两店直线距离_米"])
    f = feat[sid]
    if code in CONV:
        f["conv"] += 1; f["nearest_conv"] = min(f["nearest_conv"], d)
    elif code in SUPER:
        f["super"] += 1; f["nearest_super"] = min(f["nearest_super"], d)
    else:
        f["other"] += 1
    f["nearest_all"] = min(f["nearest_all"], d)

all_ids = set(x["本次整理门店ID"] for x in flagship)
out2 = []
for sid in all_ids:
    f = feat.get(sid, {"conv": 0, "super": 0, "other": 0,
        "nearest_all": 1e9, "nearest_conv": 1e9, "nearest_super": 1e9})
    out2.append({"store_id": sid,
        "n_convenience_500m": f["conv"], "n_supermarket_500m": f["super"],
        "n_other_500m": f["other"],
        "n_total_500m": f["conv"] + f["super"] + f["other"],
        "nearest_competitor_m": f["nearest_all"] if f["nearest_all"] < 1e9 else "",
        "nearest_convenience_m": f["nearest_conv"] if f["nearest_conv"] < 1e9 else "",
        "nearest_supermarket_m": f["nearest_super"] if f["nearest_super"] < 1e9 else ""})
print("\ncompetitor features: %d stores (file10 had %d with >=1)" % (len(out2), len(feat)))
with open(os.path.join(HERE, "competitor_features.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["store_id", "n_convenience_500m",
        "n_supermarket_500m", "n_other_500m", "n_total_500m",
        "nearest_competitor_m", "nearest_convenience_m", "nearest_supermarket_m"])
    w.writeheader(); w.writerows(out2)
print("wrote competitor_features.csv")
