"""Check Colab local disk, Google Drive storage, and remote dataset sizes."""

import json
import re
import shutil
from pathlib import Path
import urllib.request

BASE_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/"
FILES = [
    "clean_trainset_28spk_wav.zip",
    "noisy_trainset_28spk_wav.zip",
    "clean_testset_wav.zip",
    "noisy_testset_wav.zip",
    "logfiles.zip",
    "trainset_28spk_txt.zip",
    "testset_txt.zip",
    "license_text",
]


def format_bytes(b: int) -> str:
    gb = b / (1024 ** 3)
    mb = b / (1024 ** 2)
    if gb >= 1.0:
        return f"{gb:.2f} GB ({b:,} bytes)"
    return f"{mb:.2f} MB ({b:,} bytes)"


def check_disk():
    paths_to_check = ["/content", "/content/drive", "/content/drive/MyDrive"]
    disk_info = {}
    for p in paths_to_check:
        try:
            total, used, free = shutil.disk_usage(p)
            disk_info[p] = {
                "total": total,
                "used": used,
                "free": free,
                "total_str": format_bytes(total),
                "used_str": format_bytes(used),
                "free_str": format_bytes(free),
            }
        except Exception as e:
            disk_info[p] = {"error": str(e)}
    return disk_info


def get_sizes_from_html():
    html_file = Path("/content/stage2_verify/datashare_item_full.html")
    if not html_file.exists():
        # Fetch it
        url = "https://datashare.ed.ac.uk/handle/10283/2791?show=full"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    else:
        content = html_file.read_text(encoding="utf-8", errors="replace")

    # Look for bitstream rows in table or text
    # e.g. clean_trainset_28spk_wav.zip ... size
    sizes = {}
    # Let's search table rows
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', content, flags=re.S)
    for r in rows:
        for fname in FILES:
            if fname in r:
                # find numbers with Mb, Gb, or bytes
                m = re.findall(r'(\d+[\.,]?\d*)\s*(Gb|Mb|Kb|bytes|B)', r, flags=re.I)
                sizes[fname] = {"matches": m, "raw_row": re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', r)).strip()}
    return sizes


def check_remote_stream_header(fname: str):
    url = BASE_URL + fname
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    # GET request with stream, read 0 bytes, check headers
    with urllib.request.urlopen(req, timeout=30) as resp:
        cl = resp.headers.get("Content-Length")
        ct = resp.headers.get("Content-Type")
        real_url = resp.geturl()
        return {
            "url": real_url,
            "content_length": int(cl) if cl else None,
            "content_type": ct,
            "status": resp.status,
        }


def main():
    print("=== STORAGE AND REMOTE BITSTREAM INSPECTION ===")
    disks = check_disk()
    print("\n--- Disk Usage ---")
    for k, v in disks.items():
        if "error" in v:
            print(f"{k}: Error: {v['error']}")
        else:
            print(f"{k} -> Total: {v['total_str']}, Used: {v['used_str']}, Free: {v['free_str']}")

    print("\n--- Remote Edinburgh Bitstream Files (via GET Stream Headers) ---")
    file_info = {}
    total_remote_bytes = 0
    for fname in FILES:
        try:
            info = check_remote_stream_header(fname)
            cl = info["content_length"]
            if cl:
                total_remote_bytes += cl
                info["content_length_str"] = format_bytes(cl)
            file_info[fname] = info
            print(f"{fname}: {info.get('content_length_str', 'unknown')} (Type: {info.get('content_type')}) -> {info['url']}")
        except Exception as e:
            file_info[fname] = {"error": str(e)}
            print(f"{fname}: Error: {e}")

    print(f"\nTotal remote download size: {format_bytes(total_remote_bytes)}")

    html_sizes = get_sizes_from_html()
    print("\n--- Sizes from DataShare HTML Metadata ---")
    for fname, data in html_sizes.items():
        print(f"{fname}: {data['raw_row']}")

    # Check Drive target dir
    drive_data_dir = Path("/content/drive/MyDrive/Speech-Enhancement/data")
    print(f"\nTarget Drive Directory: {drive_data_dir} (Exists: {drive_data_dir.exists()})")
    if drive_data_dir.exists():
        existing = list(drive_data_dir.glob("**/*"))
        print(f"Existing items in target: {len(existing)}")
        for item in existing:
            if item.is_file():
                print(f"  {item.name}: {format_bytes(item.stat().st_size)}")

    report = {
        "disks": disks,
        "remote_files": file_info,
        "total_remote_bytes": total_remote_bytes,
        "total_remote_str": format_bytes(total_remote_bytes),
        "html_sizes": html_sizes,
    }

    out_path = Path("results/storage_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
