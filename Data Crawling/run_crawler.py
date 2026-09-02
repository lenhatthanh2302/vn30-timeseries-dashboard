#%% - Libraries
from pathlib import Path
from crawl_factory import run_crawl_job

#%% - Configuration
Date_from = "02/01/2018"
Date_to   = "31/08/2026"
Headless  = False  # Đổi thành True nếu không muốn hiện browser

#%% - Tickers VN30
Tickers = [
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS",
    "GVR", "HDB", "HPG", "LPB", "MBB", "MCH",
    "MSN", "MWG", "SAB", "SHB", "SSB", "SSI",
    "STB", "TCB", "TCX", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]

#%% - Output directory
try:
    Output_dir = Path(__file__).resolve().parents[1] / "Dataset"
except NameError:
    _cwd = Path.cwd()
    Output_dir = _cwd / "Dataset" if (_cwd / "Dataset").exists() else _cwd.parent / "Dataset"

#%% - Run
success, failed = [], []
print(f"\n{'='*55}")
print(f"Crawling {len(Tickers)} tickers | {Date_from} → {Date_to}")
print(f"Output: {Output_dir}")
print(f"{'='*55}\n")

for i, ticker in enumerate(Tickers, 1):
    print(f"[{i}/{len(Tickers)}] {ticker} — crawling...")
    try:
        run_crawl_job(ticker, Date_from, Date_to, Output_dir, Headless)
        success.append(ticker)
    except Exception as e:
        print(f"  ✘ Failed: {e}")
        failed.append(ticker)

#%% - Summary
print(f"\n{'='*55}")
print(f"Done: {len(success)} ok, {len(failed)} failed")
if success: print(f"  ✔ {success}")
if failed:  print(f"  ✘ {failed}")
print(f"{'='*55}\n")
