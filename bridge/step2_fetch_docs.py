import urllib.request
import zipfile
import json
from pathlib import Path

out_dir = Path("/content/stage2_verify")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Fetch item-level license_text
license_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/license_text"
print("=== 1. FETCHING ITEM-LEVEL LICENSE_TEXT ===")
print(f"URL: {license_url}")
req = urllib.request.Request(license_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    license_bytes = resp.read()
    license_str = license_bytes.decode("utf-8", errors="replace")
(out_dir / "license_text").write_text(license_str, encoding="utf-8")
print("--- VERBATIM LICENSE_TEXT START ---")
print(license_str.strip())
print("--- VERBATIM LICENSE_TEXT END ---\n")

# 2. Fetch official DataShare item page metadata / bitstream list and testset_txt.zip (contains README / log_testset.txt)
item_url = "https://datashare.ed.ac.uk/handle/10283/2791?show=full"
print("=== 2. FETCHING DATASHARE ITEM METADATA & TESTSET_TXT.ZIP ===")
print(f"Metadata URL: {item_url}")
req_meta = urllib.request.Request(item_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_meta, timeout=30) as resp:
    html_str = resp.read().decode("utf-8", errors="replace")
(out_dir / "datashare_item_full.html").write_text(html_str, encoding="utf-8")

# Also check testset_txt.zip (small text metadata archive ~10 KB)
txt_zip_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip"
txt_zip_path = out_dir / "testset_txt.zip"
print(f"Downloading metadata archive: {txt_zip_url}")
urllib.request.urlretrieve(txt_zip_url, str(txt_zip_path))
with zipfile.ZipFile(txt_zip_path, "r") as zf:
    print("Files inside testset_txt.zip:", zf.namelist())
    for name in zf.namelist():
        content = zf.read(name).decode("utf-8", errors="replace")
        (out_dir / Path(name).name).write_text(content, encoding="utf-8")
        lines = content.strip().splitlines()
        print(f"--- {name} (Total lines: {len(lines)}, showing first 15 lines) ---")
        for line in lines[:15]:
            print("  ", line)
print()
