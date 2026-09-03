# VN30 Stock Price Forecasting — Big Data Pipeline

Dự báo giá đóng cửa 30 cổ phiếu VN30 (2018–2026) sử dụng PySpark kết hợp 5 mô hình:
ARIMA, Holt-Winters, Linear Regression, Random Forest và GBT.

Kết quả được trực quan hóa qua dashboard Streamlit với 3 tab:
Phân tích từng ticker, so sánh 30 tickers và upload phân tích dữ liệu mới.

---

## Mục lục

1. [Cấu trúc Project](#1-cấu-trúc-project)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Cấu hình PySpark trên Windows](#4-cấu-hình-pyspark-trên-windows)
5. [Cập nhật Python path trong pipeline.py](#5-cập-nhật-python-path-trong-pipelinepy)
6. [Quy trình chạy](#6-quy-trình-chạy)
7. [Xử lý lỗi thường gặp](#7-xử-lý-lỗi-thường-gặp)

---

## 1. Cấu trúc Project

```
Big Data_Nhóm Gia đình Haha/
├── Data Crawling/
│   ├── crawl_factory.py       # Logic crawl + helper functions
│   ├── run_crawler.py         # Config và entry point để crawl
│   └── README_CRAWL.md        # Tài liệu kỹ thuật chi tiết về crawler
├── Dataset/
│   ├── ACB_2018_2026.csv      # 30 file CSV (1 file / ticker)
│   ├── BID_2018_2026.csv
│   ├── ... (28 file còn lại)
│   └── results/
│       ├── predictions.csv    # Dự báo từ 5 models (Streamlit đọc)
│       ├── predictions.parquet
│       └── metrics.csv        # MAE, RMSE, MAPE, R² per ticker×model
├── Model/
│   ├── pipeline.py            # PySpark pipeline (chạy offline)
│   └── README_MODELS.md       # Tài liệu kỹ thuật chi tiết về pipeline
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Dependencies cho Streamlit
├── pyproject.toml
└── README.md                  # File này
```

> **Lưu ý về workflow:** `pipeline.py` chạy offline trên máy local để tạo
> `Dataset/results/`. Sau đó `app.py` chỉ đọc các file kết quả đó — Streamlit
> không chạy PySpark trực tiếp.

---

## 2. Yêu cầu hệ thống

Cài đặt đầy đủ các thành phần dưới đây **trước khi** tiến hành bất kỳ bước nào khác.

### 2.1 Python 3.12

Tải và cài từ [python.org](https://www.python.org/downloads/).
Chọn đúng **Python 3.12.x** — PySpark có thể không tương thích với Python 3.13+.

Trong lúc cài, **tick vào "Add Python to PATH"**.

Kiểm tra sau khi cài:

```
python --version
# Python 3.12.x
```

### 2.2 Java JDK 11 hoặc 17

PySpark bắt buộc phải có Java. Không có Java thì `import pyspark` sẽ báo lỗi ngay.

Tải **Eclipse Temurin (Adoptium)** — phiên bản JDK 17 LTS:
[adoptium.net/temurin/releases](https://adoptium.net/temurin/releases/)

Chọn: `Windows` → `x64` → `JDK` → `17` → `.msi` → tải về và cài.

Trong lúc cài wizard, bật tùy chọn **"Set JAVA_HOME variable"** nếu có.

Kiểm tra sau khi cài:

```
java -version
# openjdk version "17.x.x" ...
```

Nếu lệnh trên không nhận ra, cần set thủ công (xem bên dưới).

**Set JAVA_HOME thủ công (nếu cần):**

1. Mở **Start** → tìm "Environment Variables" → mở "Edit the system environment variables"
2. Click **"Environment Variables..."**
3. Trong phần "System variables", click **New**:
   - Variable name: `JAVA_HOME`
   - Variable value: đường dẫn đến folder JDK, ví dụ `C:\Program Files\Eclipse Adoptium\jdk-17.x.x.x-hotspot`
4. Tìm biến `Path` trong "System variables" → click **Edit** → click **New** → thêm `%JAVA_HOME%\bin`
5. Click OK tất cả, rồi **khởi động lại terminal**

### 2.3 Google Chrome

Cần để chạy crawler (Selenium). Selenium 4 tự quản lý ChromeDriver — không cần cài thêm.

Tải từ [google.com/chrome](https://www.google.com/chrome/) nếu chưa có.

### 2.4 Git *(tùy chọn)*

Nếu clone từ GitHub, cần cài Git:
[git-scm.com/downloads](https://git-scm.com/downloads)

---

## 3. Cài đặt môi trường

> **Quan trọng:** Không tạo virtual environment **bên trong** folder project vì
> đường dẫn quá dài sẽ gây lỗi `WinError 206` khi cài `torch` và một số package
> khác trên Windows. Tạo venv ở **ngoài** project, ví dụ `C:\venvs\bigdata`.

### Bước 1 — Lấy source code

**Clone từ GitHub:**

```bash
git clone <URL_repo_GitHub>
cd "Big Data_Nhóm Gia đình Haha"
```

**Hoặc giải nén file ZIP** nếu được chia sẻ qua Drive/USB, rồi mở folder đó.

### Bước 2 — Tạo virtual environment

Mở **Command Prompt** (không phải PowerShell) và chạy:

```
python -m venv C:\venvs\bigdata
```

Kích hoạt:

```
C:\venvs\bigdata\Scripts\activate
```

Dấu nhắc lệnh sẽ thay đổi thành `(bigdata) C:\...>` — tức là venv đã hoạt động.

### Bước 3 — Cài Streamlit dependencies

```
pip install -r requirements.txt
```

File `requirements.txt` bao gồm: `streamlit`, `pandas`, `numpy`, `plotly`,
`statsmodels`, `scikit-learn`, `pyarrow`.

### Bước 4 — Cài PySpark và modeling dependencies

```
pip install pyspark==3.5.1 pmdarima selenium
```

Kiểm tra PySpark đã cài đúng:

```
python -c "import pyspark; print(pyspark.__version__)"
# 3.5.1
```

### Bước 5 — Kiểm tra toàn bộ

```
python -c "import pyspark, pmdarima, streamlit, plotly, statsmodels; print('OK')"
# OK
```

---

## 4. Cấu hình PySpark trên Windows

PySpark trên Windows cần thêm `winutils.exe` — một binary nhỏ giúp Spark thực hiện
các thao tác file system. Thiếu file này sẽ gây lỗi:
`Could not locate executable null\bin\winutils.exe in the Hadoop binaries`.

### Bước 1 — Tải winutils.exe

Tìm file `winutils.exe` cho **Hadoop 3.x** từ repo cộng đồng trên GitHub
(search: `winutils hadoop 3 windows`). Tải file `winutils.exe` về.

### Bước 2 — Đặt file vào đúng chỗ

Tạo folder:

```
C:\hadoop\bin\
```

Đặt file `winutils.exe` vừa tải vào folder `C:\hadoop\bin\`.

Cấu trúc kết quả:

```
C:\
└── hadoop\
    └── bin\
        └── winutils.exe
```

### Bước 3 — Set biến môi trường HADOOP_HOME

1. Mở **Start** → "Environment Variables" → "Edit the system environment variables"
2. Click **"Environment Variables..."**
3. Trong "System variables", click **New**:
   - Variable name: `HADOOP_HOME`
   - Variable value: `C:\hadoop`
4. Tìm `Path` → **Edit** → **New** → thêm `C:\hadoop\bin`
5. Click OK tất cả → **khởi động lại terminal**

### Bước 4 — Kiểm tra

Mở terminal mới, kích hoạt venv, rồi chạy:

```
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.getOrCreate(); print('Spark OK'); spark.stop()"
```

Nếu không có lỗi đỏ và in ra `Spark OK` thì xong.

---

## 5. Cập nhật Python path trong `pipeline.py`

`pipeline.py` có một dòng hardcode đường dẫn Python để PySpark worker dùng đúng
Python của venv. **Bắt buộc phải cập nhật dòng này trên mỗi máy mới.**

Mở `Model/pipeline.py`, tìm đoạn sau (khoảng dòng 45):

```python
_py = r'C:\Users\ThanhT\AppData\Local\Programs\Python\Python312\python.exe'
```

Thay bằng đường dẫn đến `python.exe` **trong venv của bạn**.

Nếu bạn tạo venv tại `C:\venvs\bigdata` (theo hướng dẫn ở trên), sửa thành:

```python
_py = r'C:\venvs\bigdata\Scripts\python.exe'
```

**Cách tìm đường dẫn chính xác trên máy của bạn:**

Kích hoạt venv rồi chạy:

```
python -c "import sys; print(sys.executable)"
```

Copy kết quả in ra và dán vào dòng `_py = r'...'` trong `pipeline.py`.

---

## 6. Quy trình chạy

Có 3 bước, chạy theo thứ tự. Nếu bạn nhận được project từ người khác và
**folder `Dataset/` đã có đủ 30 file CSV và folder `results/` đã có sẵn**,
bạn có thể bỏ qua Bước 1 và 2, chạy thẳng Bước 3.

---

### Bước 1 — Crawl dữ liệu (nếu Dataset/ chưa có CSV)

Di chuyển vào folder `Data Crawling`:

```
cd "Data Crawling"
```

Mở `run_crawler.py` và kiểm tra cấu hình:

```python
Date_from = "02/01/2018"
Date_to   = "31/08/2026"
Headless  = False          # False = hiện browser, True = ẩn browser
```

Chạy:

```
python run_crawler.py
```

Quá trình crawl **lần lượt từng ticker** (30 tickers, mỗi ticker ~3–10 phút tùy tốc độ mạng).
Tổng thời gian ước tính: **2–4 giờ**.

Mỗi ticker sẽ được lưu thành 1 file CSV vào `Dataset/`:

```
Dataset/
├── ACB_2018_2026.csv
├── BID_2018_2026.csv
└── ... (30 files)
```

> **Nếu bị ngắt giữa chừng:** Chạy lại `run_crawler.py` bình thường.
> Crawler tự động bỏ qua các ticker đã có file CSV, chỉ crawl tiếp các ticker còn lại.

---

### Bước 2 — Chạy PySpark pipeline (nếu results/ chưa có)

Kích hoạt venv nếu chưa:

```
C:\venvs\bigdata\Scripts\activate
```

Di chuyển vào folder `Model`:

```
cd ..\Model
```

Chạy pipeline:

```
python pipeline.py
```

Pipeline sẽ lần lượt:
- Load và làm sạch 30 CSV bằng PySpark
- Tạo features (lag, rolling mean/std, dow, month)
- Chia train/test (80/20, theo thứ tự thời gian)
- Huấn luyện Linear Regression, Random Forest, GBT bằng PySpark MLlib
- Huấn luyện ARIMA và Holt-Winters per-ticker bằng statsmodels
- Tính MAE, RMSE, MAPE, R² cho từng model × ticker
- Lưu kết quả vào `Dataset/results/`

Thời gian ước tính: **30–90 phút** tùy cấu hình máy.

Output khi hoàn thành:

```
Loaded 61,000+ rows across 30 tickers
Train: 49,xxx rows | Test: 12,xxx rows
Linear Regression ✔
Random Forest ✔
GBT ✔
ARIMA ✔
Holt-Winters ✔

Results saved → .../Dataset/results
=================================================================
Mean metrics across all tickers (sorted by RMSE)
=================================================================
...
```

---

### Bước 3 — Chạy Streamlit dashboard

Kích hoạt venv nếu chưa:

```
C:\venvs\bigdata\Scripts\activate
```

Di chuyển về **root folder** của project (nơi chứa `app.py`):

```
cd ..
```

Chạy:

```
streamlit run app.py
```

Streamlit sẽ tự mở trình duyệt tại `http://localhost:8501`.
Nếu không tự mở, copy URL đó dán vào Chrome.

Dashboard có 3 tab:
- **Tab 1 — Phân tích theo Ticker:** Chọn 1 trong 30 tickers để xem overview, rolling stats, kiểm định ADF/KPSS, ACF/PACF, decomposition và kết quả dự báo của 5 models.
- **Tab 2 — So sánh 30 Tickers:** Bảng tổng hợp, biểu đồ bar RMSE/MAE/MAPE/R², heatmap và bảng best model per ticker.
- **Tab 3 — Upload & Analyze:** Upload CSV của bất kỳ cổ phiếu nào (cần cột `Date` định dạng `DD/MM/YYYY` và `Close_price`). Hỗ trợ 2 models: Linear Trend và Holt-Winters.

Để dừng dashboard: nhấn `Ctrl + C` trong terminal.

---

## 7. Xử lý lỗi thường gặp

### `JAVA_HOME is not set`

PySpark không tìm thấy Java. Kiểm tra lại Bước 2.2 — đảm bảo biến `JAVA_HOME`
đã được set và terminal đã được khởi động lại sau khi set.

```
echo %JAVA_HOME%
# Phải in ra đường dẫn đến JDK, ví dụ: C:\Program Files\Eclipse Adoptium\jdk-17...
```

---

### `Could not locate executable null\bin\winutils.exe`

Chưa cài `winutils.exe` hoặc biến `HADOOP_HOME` chưa đúng. Kiểm tra lại Mục 4.

```
echo %HADOOP_HOME%
# Phải in ra: C:\hadoop
```

---

### `Error in PYSPARK_PYTHON` hoặc Python worker crash

Dòng `_py` trong `pipeline.py` trỏ sai đường dẫn. Kiểm tra lại Mục 5.

---

### `ModuleNotFoundError: No module named 'pmdarima'` (hoặc module khác)

Package chưa được cài vào đúng venv. Đảm bảo venv đã được kích hoạt
trước khi `pip install`:

```
C:\venvs\bigdata\Scripts\activate
pip install pyspark==3.5.1 pmdarima selenium
```

---

### Crawler bị lỗi giữa chừng

Chạy lại `python run_crawler.py` — các ticker đã có CSV sẽ bị bỏ qua tự động.
Nếu 1 ticker liên tục thất bại, thử đổi `Headless = False` để quan sát trực tiếp
browser đang làm gì.

---

### Streamlit báo `FileNotFoundError: predictions.csv`

Folder `Dataset/results/` chưa có file kết quả. Cần chạy `pipeline.py` (Bước 2) trước.

---

## Nhóm thực hiện

**Nhóm Gia đình Haha — Môn Nghiên cứu Dữ liệu lớn và Ứng dụng trong Kinh doanh (253MIE400801)**

| MSHV | Họ và tên |
|---|---|
| C25611254 | Lê Nhật Thanh |
| C25611253 | Lê Ngọc Phú |
| C25611251 | Huỳnh Trúc Ngân |
| C25611258 | Phạm Thị Thúy Kiều |
| C25611256 | Trịnh Quang Tân |

GVHD: TS. Nguyễn Thôn Dã
