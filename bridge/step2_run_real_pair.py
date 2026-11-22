import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

out_dir = Path("/content/stage2_verify")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Parse DataShare HTML for official Dublin Core metadata and bitstream list
html_str = (out_dir / "datashare_item_full.html").read_text(encoding="utf-8", errors="replace")
print("=== 1. OFFICIAL DATASHARE ITEM METADATA (HANDLE 10283/2791) ===")
meta_matches = re.findall(r'<meta name="(DC\.[^"]+|DCTERMS\.[^"]+)" content="([^"]+)"', html_str)
for k, v in meta_matches:
    print(f"  {k}: {v}")

bitstreams = re.findall(r'href="(/bitstream/handle/10283/2791/[^"?]+)', html_str)
unique_bitstreams = sorted(set(bitstreams))
print("\n=== 2. OFFICIAL BITSTREAMS LISTED ON HANDLE 10283/2791 ===")
for b in unique_bitstreams:
    print("  https://datashare.ed.ac.uk" + b)

# 3. Download logfiles.zip if present to inspect log_testset.txt / README
for b in unique_bitstreams:
    if "log" in b.lower() or "readme" in b.lower():
        url = "https://datashare.ed.ac.uk" + b
        fname = out_dir / Path(b).name
        print(f"\nDownloading documentation bitstream: {url}")
        urllib.request.urlretrieve(url, str(fname))
        if fname.suffix == ".zip":
            with zipfile.ZipFile(fname, "r") as zf:
                print("  Contents:", zf.namelist())
                for member in zf.namelist():
                    if not member.endswith("/") and "__MACOSX" not in member:
                        txt = zf.read(member).decode("utf-8", errors="replace")
                        lines = txt.strip().splitlines()
                        print(f"  --- {member} ({len(lines)} lines, first 10 lines) ---")
                        for l in lines[:10]:
                            print("   ", l)
        else:
            txt = fname.read_text(encoding="utf-8", errors="replace")
            print(txt[:2000])

# 4. Extract ONLY ONE real clean/noisy speech pair (p232_001.wav) from testset ZIPs for Stage 2 Smoke Test
clean_zip_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
noisy_zip_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
clean_zip_path = out_dir / "clean_testset_wav.zip"
noisy_zip_path = out_dir / "noisy_testset_wav.zip"

print("\n=== 3. EXTRACTING SINGLE REAL AUDIO PAIR (p232_001.wav) FOR STAGE 2 SMOKE TEST ===")
if not clean_zip_path.exists():
    urllib.request.urlretrieve(clean_zip_url, str(clean_zip_path))
if not noisy_zip_path.exists():
    urllib.request.urlretrieve(noisy_zip_url, str(noisy_zip_path))

clean_wav_out = out_dir / "clean_p232_001.wav"
noisy_wav_out = out_dir / "noisy_p232_001.wav"

with zipfile.ZipFile(clean_zip_path, "r") as zf:
    wav_members = [m for m in zf.namelist() if m.endswith("p232_001.wav") and "__MACOSX" not in m]
    assert len(wav_members) == 1, f"Expected 1 match, got {wav_members}"
    clean_wav_out.write_bytes(zf.read(wav_members[0]))

with zipfile.ZipFile(noisy_zip_path, "r") as zf:
    wav_members = [m for m in zf.namelist() if m.endswith("p232_001.wav") and "__MACOSX" not in m]
    assert len(wav_members) == 1, f"Expected 1 match, got {wav_members}"
    noisy_wav_out.write_bytes(zf.read(wav_members[0]))

# Delete the temporary ZIPs immediately so we do not keep full archives on disk
clean_zip_path.unlink(missing_ok=True)
noisy_zip_path.unlink(missing_ok=True)

print(f"Extracted single pair: {clean_wav_out} ({clean_wav_out.stat().st_size} bytes) and {noisy_wav_out} ({noisy_wav_out.stat().st_size} bytes)")
print("Transcript for p232_001.txt:", (out_dir / "p232_001.txt").read_text(encoding="utf-8").strip())

# 5. Run Stage 2 Real-Audio Smoke Test via scripts/04_smoke_test.py
print("\n=== 4. RUNNING SCRIPTS/04_SMOKE_TEST.PY WITH REAL AUDIO PAIR (p232_001.wav) ===")
res = subprocess.run(
    [
        "python",
        "scripts/04_smoke_test.py",
        "--clean_wav",
        str(clean_wav_out),
        "--noisy_wav",
        str(noisy_wav_out),
    ],
    capture_output=True,
    text=True,
)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
assert res.returncode == 0, f"Smoke test failed with exit code {res.returncode}"
