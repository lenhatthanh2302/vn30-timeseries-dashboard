# Vietnam Stock Market Data Crawler

A web scraping tool built with Python and Selenium that pulls
historical trading data from [Vietstock Finance](https://finance.vietstock.vn)
and stores it as structured CSV files, ready for PySpark analysis.

---

## Why this exists

Vietstock does not offer a free data export feature for long historical
ranges. Downloading data manually — day by day or page by page — is
tedious and error-prone. This tool automates the entire process: open
the page, set the date range, load every record, and save everything
to a clean CSV file, all without any manual interaction.

---

## Project Structure

```
Big Data_Nhóm Gia đình Haha/
├── Data Crawling/
│   ├── crawl_factory.py   # All crawling logic and helper functions
│   └── run_crawler.py     # Your configuration and entry point
├── Dataset/               # CSV output files land here
│   └── results/           # Pre-computed model outputs (for Streamlit)
├── Model/
│   └── pipeline.py        # PySpark modeling pipeline
└── app.py                 # Streamlit dashboard
```

The split is intentional. `crawl_factory.py` is the machinery —
you should rarely need to touch it. `run_crawler.py` is the only
file you interact with day to day.

---

## 1. Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install selenium pandas
```

Selenium 4 manages ChromeDriver automatically via **Selenium Manager** —
no manual ChromeDriver download needed. Just make sure Google Chrome
is installed and up to date.

---

### 2. Setup & Run `run_crawler.py`

**Step 1 — Open `run_crawler.py` and fill in your parameters:**

```python
Date_from = "02/01/2018"    # Start of the data range  (DD/MM/YYYY)
Date_to   = "31/08/2026"    # End of the data range    (DD/MM/YYYY)
Headless  = False           # True = run without opening a browser window

Tickers = [
    "FPT", "VIC", "HPG",
]                           # The tickers you want to crawl
```

**Step 2 — Run:**

```bash
python run_crawler.py
```

**Step 3 — File location:**

```
Dataset/
├── FPT_2018_2026.csv
├── VIC_2018_2026.csv
└── HPG_2018_2026.csv
```

One CSV per ticker, named by symbol and year range.
If a file already exists, that ticker is skipped automatically.

---

## Configuration Reference (`run_crawler.py`)

| Parameter | What it controls |
|---|---|
| `Date_from` | Start date of the historical range, in DD/MM/YYYY |
| `Date_to` | End date of the historical range, in DD/MM/YYYY |
| `Tickers` | List of stock symbols to crawl — add or remove freely |
| `Headless` | `False` = visible browser, `True` = headless (no window) |

The URL is built from the ticker automatically:

```
"FPT"  →  https://finance.vietstock.vn/FPT/thong-ke-giao-dich.htm
"VIC"  →  https://finance.vietstock.vn/VIC/thong-ke-giao-dich.htm
```

When crawling multiple tickers, each one runs sequentially.
If one fails mid-way, the rest still continue. A summary is printed at the end:

```
=======================================================
Done: 3 ok, 0 failed
  ✔ ['FPT', 'VIC', 'HPG']
=======================================================
```

---

## 3. Understanding `crawl_factory.py`

### `create_driver(headless=False)`

Opens a Chrome window with some settings applied:
notifications, infobars, and extensions are disabled; the window starts
maximized; noisy console logs from Chrome internals are suppressed.
Pass `headless=True` to run without a visible browser window.

---

### `robust_click(driver, element)`

A normal Selenium `.click()` breaks more often than expected —
an ad overlay, a fixed header, or an element slightly off-screen can
all cause it to fail silently or throw an exception.
This function tries three approaches in sequence and stops at the first one that works:

```
1. Standard .click()
      ↓ blocked by overlay or not interactable?
2. JavaScript click  →  driver.execute_script("arguments[0].click()")
      ↓ still failing?
3. ActionChains      →  move mouse to element, then click
```

Returns `True` if any attempt succeeded, `False` if all three failed.

---

### `dismiss_ad(driver, timeout)`

Vietstock occasionally serves a banner ad that sits at the bottom of
the screen and blocks the **"Load More"** button. This function detects
and closes it.

The first thing it does is a near-instant DOM check:

```python
if not driver.find_elements(*Ad_Wrapper):
    return False   # No ad present — exit immediately
```

This matters because `dismiss_ad()` is called repeatedly inside the loading
loop. Without this check, the function would wait out a full timeout on every
single click. With it, the no-ad case costs almost nothing.

If the wrapper is found, it then waits for the close button to become clickable,
scrolls it into the center of the viewport, and clicks it via `robust_click()`.
Afterwards, it waits for the banner to disappear before returning.

---

### `set_date_input(driver, wait, locator, date_str)`

Inputs a date value into a date field using Selenium:
clicks the field, selects all existing text, deletes it, types the new date,
and confirms with Enter.

---

### `load_all_rows(driver)`

Vietstock paginates its table behind a "Load More" button rather than
across multiple pages. To get the full history, you have to keep clicking
it until it disappears.

Several targeted optimizations are in place:

**Faster row detection** — The row count is read directly from the browser
using a small JavaScript snippet instead of collecting all row elements
into a Python list on each iteration.

**Faster change detection** — `WebDriverWait` is initialized with
`poll_frequency=0.2`, checking for new rows every 200 ms instead of
the default 500 ms.

**Smarter scrolling** — The button is scrolled into view only when its
position on the page has shifted by more than 50 pixels, meaning new rows
pushed it further down.

**Ad handling without slowdown** — `dismiss_ad()` is called between clicks
but returns almost instantly when no ad is present.

**Failure guard** — If `robust_click()` fails five times in a row, the loop
stops rather than hanging indefinitely.

Returns `(click_count, final_row_count)`.

---

### `extract_table_data(driver)`

Extracts all table rows in a **single JavaScript call** — the entire table
is read from the DOM in one round trip rather than iterating over each row
from Python. This is significantly faster once thousands of rows are loaded.

The JS snippet filters out rows that do not have exactly 18 cells, maps
each cell's text to the corresponding column header, and replaces `#####`
rendering artifacts with `N/A`.

Returns a list of dictionaries, one per valid data row.

---

### `save_to_csv(records, ticker, date_from, date_to, output_dir)`

Converts the list of row dictionaries into a pandas DataFrame, drops
any exact duplicate rows, and writes the result to a `.csv` file.

The filename is derived from the ticker and the year portion of the date range:

```
date_from = "02/01/2018"  →  year = "2018"
date_to   = "31/08/2026"  →  year = "2026"
filename  = "FPT_2018_2026.csv"
```

If the `Dataset/` folder does not exist yet, it is created automatically.

---

### `run_crawl_job(ticker, date_from, date_to, output_dir, headless)`

The top-level function that `run_crawler.py` calls. It wires all the
functions above into a single sequential pipeline:

```
0. Check if output file already exists → skip if yes
1. Open the ticker's page
2. Wait for initial rows to appear
3. Close ad (if any)
4. Set the From Date  →  check for ad
5. Set the To Date    →  check for ad
6. Wait for table to reload
7. Load all rows (keep clicking "Load More")
8. Extract table data (single JS call)
9. Save to CSV
10. Close the browser
```

Each ticker gets its own browser session. A crash or timeout on one
ticker will not affect the others in the batch.

---

## Data Columns

Each CSV file contains the following columns:

| Column | Description |
|---|---|
| `No` | Row index |
| `Date` | Trading date (DD/MM/YYYY) |
| `Ticker` | Stock symbol |
| `Reference` | Reference price for the session |
| `Open_price` | Opening price |
| `Close_price` | Closing price |
| `Highest_price` | Intraday high |
| `Lowest_price` | Intraday low |
| `Average` | Volume-weighted average price |
| `Change` | Absolute price change from reference |
| `Change_percent` | Percentage price change from reference |
| `Matched_volume` | Order-matched trading volume |
| `Matched_value` | Order-matched trading value |
| `Negotiated_volume` | Put-through (negotiated deal) volume |
| `Negotiated_value` | Put-through (negotiated deal) value |
| `Total_volume` | Combined matched + negotiated volume |
| `Total_value` | Combined matched + negotiated value |
| `Market_cap` | Market capitalization at close |

---

## Dependencies

| Package | Purpose |
|---|---|
| `selenium` | Browser automation and page interaction |
| `pandas` | Data structuring and CSV export |
| `ChromeDriver` | Managed automatically by Selenium 4 |
