#!/usr/bin/env python3

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path


BASE = "https://moon.bao.ac.cn/moon-admin/client/science"
OUT = Path(__file__).parent
opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)
query = urllib.parse.urlencode(
    {"pageNum": 1, "pageSize": 1000, "taskName": "CE4", "loadName": "LFRS", "levelName": "2C"}
)
products = json.load(urllib.request.urlopen(f"{BASE}/dataInfoList?{query}"))["rows"]

if sys.argv[1:] == ["demo"]:
    products = [p for p in products if "20190103162200_20190103174300_0001_B." in p["name"]]
elif sys.argv[1:]:
    raise SystemExit("usage: download_ce4_lfrs_2c.py [demo]")
for product in products:
    path = OUT / product["name"]
    size = int(product["dataSize"])
    if path.exists() and path.stat().st_size == size:
        print("skip", path.name)
        continue
    url = json.load(
        urllib.request.urlopen(f"{BASE}/dataInfo/getAnnexZip/{product['dataInfoId']}")
    )["data"]
    print("download", path.name)
    urllib.request.urlretrieve(url, path)
