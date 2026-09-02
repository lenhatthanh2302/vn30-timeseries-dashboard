#%% - Libraries
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, ElementClickInterceptedException,
    ElementNotInteractableException, StaleElementReferenceException)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
import pandas as pd
import os

#%% - Locators and Headers
Ad_Wrapper   = (By.CSS_SELECTOR, "div.ats-overlay-bottom-wrapper-rendered")
Ad_Close_Btn = (By.CSS_SELECTOR, "button.ats-overlay-bottom-close-button")
Date_From    = (By.XPATH, '//*[@id="txtFromTradeDate"]')
Date_To      = (By.XPATH, '//*[@id="txtToTradeDate"]')
LoadMore_Btn = (By.XPATH, '//*[@id="trading-list"]/div/div[2]/div[2]/span')
TBody        = (By.XPATH, '//*[@id="trading-list"]/div/div[2]/div[1]/table/tbody')
TRow         = (By.XPATH, '//*[@id="trading-list"]/div/div[2]/div[1]/table/tbody/tr')

HEADERS = [
    "No", "Date", "Ticker", "Reference", "Open_price", "Close_price",
    "Highest_price", "Lowest_price", "Average",
    "Change", "Change_percent", "Matched_volume", "Matched_value",
    "Negotiated_volume", "Negotiated_value",
    "Total_volume", "Total_value", "Market_cap"
]

#%% - Build output path
def _build_path(ticker, date_from, date_to, output_dir=None):
    if output_dir is None:
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path(os.getcwd()).resolve()
        output_dir = root / "Dataset"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yyyy_f = date_from.split("/")[2]
    yyyy_t = date_to.split("/")[2]
    return output_dir / f"{ticker}_{yyyy_f}_{yyyy_t}.csv"

#%% - Setup browser
def create_driver(headless=False):
    opts = Options()
    opts.page_load_strategy = "none"  # return immediately; we wait for TRow ourselves
    opts.add_argument("--disable-notifications")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    if headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=opts)

#%% - Robust click
def robust_click(driver, element):
    strategies = [
        lambda: element.click(),
        lambda: driver.execute_script("arguments[0].click();", element),
        lambda: ActionChains(driver).move_to_element(element).click().perform(),
    ]
    for fn in strategies:
        try:
            fn()
            return True
        except Exception:
            pass
    return False

#%% - Dismiss ads
def dismiss_ad(driver, timeout=8):
    if not driver.find_elements(*Ad_Wrapper):
        return False
    try:
        btn = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable(Ad_Close_Btn))
    except TimeoutException:
        try:
            btn = driver.find_element(*Ad_Close_Btn)
        except NoSuchElementException:
            return False
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    robust_click(driver, btn)
    try:
        WebDriverWait(driver, 5).until(EC.invisibility_of_element_located(Ad_Wrapper))
    except TimeoutException:
        pass
    return True

#%% - Set date input
def set_date_input(driver, wait, locator, date_str):
    el = wait.until(EC.element_to_be_clickable(locator))
    robust_click(driver, el)
    el.send_keys(Keys.CONTROL + "a")
    el.send_keys(Keys.DELETE)
    el.clear()
    el.send_keys(date_str)
    el.send_keys(Keys.ENTER)

#%% - Load all rows
def load_all_rows(driver):
    fast_wait = WebDriverWait(driver, 20, poll_frequency=0.2)
    click_count = 0
    fail_streak = 0
    last_y      = None

    count_js = """
        var t = document.querySelector(
            '#trading-list > div > div:nth-child(2) > div:first-child > table > tbody'
        );
        return t ? t.querySelectorAll('tr').length : 0;
    """

    while True:
        btns = driver.find_elements(*LoadMore_Btn)
        if not btns or not btns[0].is_displayed():
            break

        btn = btns[0]
        y = btn.location.get("y", 0)
        if last_y is None or abs(y - last_y) > 50:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            last_y = y

        dismiss_ad(driver, timeout=2)
        prev = driver.execute_script(count_js)

        if not robust_click(driver, btn):
            fail_streak += 1
            if fail_streak >= 5:
                break
            continue

        click_count += 1
        fail_streak  = 0

        try:
            fast_wait.until(lambda d: d.execute_script(count_js) > prev)
        except TimeoutException:
            break

    return click_count, driver.execute_script(count_js)

#%% - Extract table data (single JS call — avoids thousands of WebDriver round trips)
def extract_table_data(driver):
    rows = driver.execute_script("""
        var tbody = document.querySelector(
            '#trading-list > div > div:nth-child(2) > div:first-child > table > tbody'
        );
        if (!tbody) return [];
        return Array.from(tbody.querySelectorAll('tr')).map(function(row) {
            return Array.from(row.querySelectorAll('td')).map(function(c) {
                return c.textContent.trim().replace('#####', 'N/A');
            });
        }).filter(function(r) { return r.length === 18; });
    """)
    return [dict(zip(HEADERS, row)) for row in (rows or [])]

#%% - Save to CSV
def save_to_csv(records, ticker, date_from, date_to, output_dir=None):
    if not records:
        return None
    df = pd.DataFrame(records, columns=HEADERS)
    df.drop_duplicates(inplace=True)
    path = _build_path(ticker, date_from, date_to, output_dir)
    df.to_csv(path, index=False)
    return path

#%% - Master runner
def run_crawl_job(ticker, date_from, date_to, output_dir=None, headless=False):
    url = f"https://finance.vietstock.vn/{ticker}/thong-ke-giao-dich.htm"

    path = _build_path(ticker, date_from, date_to, output_dir)
    if path.exists():
        print(f"  [Skip] {path.name} already exists")
        return

    driver = create_driver(headless=headless)
    wait      = WebDriverWait(driver, 15, poll_frequency=0.2)
    wait_long = WebDriverWait(driver, 120, poll_frequency=0.5)

    try:
        driver.get(url)
        wait_long.until(EC.presence_of_element_located(TRow))
        dismiss_ad(driver, timeout=10)

        set_date_input(driver, wait, Date_From, date_from)
        dismiss_ad(driver, timeout=5)
        set_date_input(driver, wait, Date_To, date_to)
        dismiss_ad(driver, timeout=5)
        wait_long.until(EC.presence_of_element_located(TRow))

        clicks, _ = load_all_rows(driver)
        records   = extract_table_data(driver)
        save_to_csv(records, ticker, date_from, date_to, output_dir)

        print(f"  ✔ {len(records)} rows | {clicks} clicks → {path.name}")

    except Exception as e:
        print(f"  ✘ Fatal: {type(e).__name__}: {e}")
        raise

    finally:
        driver.quit()
