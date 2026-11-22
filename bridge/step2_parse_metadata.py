import re
import urllib.request
import zipfile
from pathlib import Path

out_dir = Path("/content/stage2_verify")
html_str = (out_dir / "datashare_item_full.html").read_text(encoding="utf-8", errors="replace")

print("=== 1. DUBLIN CORE METADATA TABLE (10283/2791?show=full) ===")
rows = re.findall(r'<tr class="ds-table-row[^"]*">\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>(.*?)</td>', html_str, flags=re.S)
for k, v in rows:
    clean_v = re.sub(r'<[^>]+>', ' ', v).strip()
    clean_v = re.sub(r'\s+', ' ', clean_v)
    print(f"  {k.strip()}: {clean_v}")

# Download logfiles.zip to inspect log_testset.txt and log_trainset_28spk.txt
log_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/logfiles.zip"
log_zip = out_dir / "logfiles.zip"
print(f"\n=== 2. FETCHING OFFICIAL LOGFILES.ZIP ({log_url}) ===")
urllib.request.urlretrieve(log_url, str(log_zip))
with zipfile.ZipFile(log_zip, "r") as zf:
    print("Files inside logfiles.zip:", zf.namelist())
    for m in zf.namelist():
        if not m.endswith("/") and "__MACOSX" not in m:
            txt = zf.read(m).decode("utf-8", errors="replace")
            lines = txt.strip().splitlines()
            print(f"\n--- {m} (Total lines: {len(lines)}, first 10 lines) ---")
            for l in lines[:10]:
                print("  ", l)

# Save verbatim license_text into bridge/license_text.txt so local workspace can inspect it directly
license_local = Path("bridge/verified_license_text.txt")
license_local.write_text((out_dir / "license_text").read_text(encoding="utf-8"), encoding="utf-8")
