#!/usr/bin/env python3
"""Incrementally add market-relevant NYT Archive headlines to the headline CSV.

The NYT Archive API returns every article in a requested month. This script
filters the response in memory and persists only articles relevant to the
S&P 500 or economic factors likely to move the broad U.S. equity market.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HEADLINES_CSV = BASE_DIR / "data" / "sp500 headlines 2008 to 2024.csv"
DEFAULT_PRICES_CSV = BASE_DIR / "data" / "yfinance_sp500.csv"
DEFAULT_END_DATE = date(2025, 12, 31)
ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

DIRECT_MARKET_RE = re.compile(
    r"\b(?:s\s*&\s*p\s*500|standard\s+(?:&|and)\s+poor'?s|spx|"
    r"stock\s+market|stock\s+markets|u\.?s\.?\s+stocks?|american\s+stocks?|"
    r"wall\s+street|dow(?:\s+jones)?|nasdaq|equities|equity\s+market|"
    r"index\s+futures|stock\s+futures|market\s+(?:rally|selloff|rout|correction))\b",
    re.IGNORECASE,
)
MACRO_RE = re.compile(
    r"\b(?:federal\s+reserve|the\s+fed|fed\s+(?:rate|rates|policy|meeting)|"
    r"interest\s+rates?|rate\s+(?:cut|cuts|hike|hikes)|jerome\s+powell|"
    r"inflation|consumer\s+price(?:s|\s+index)?|cpi|jobs?\s+report|"
    r"nonfarm\s+payrolls?|unemployment|recession|economic\s+growth|"
    r"gross\s+domestic\s+product|gdp|treasury\s+yields?|bond\s+yields?|"
    r"debt\s+ceiling|government\s+shutdown|tariffs?)\b",
    re.IGNORECASE,
)
MARKET_IMPACT_RE = re.compile(
    r"\b(?:market|markets|stocks?|shares?|investors?|economy|economic|"
    r"rates?|yields?|inflation|trade|wall\s+street|futures)\b",
    re.IGNORECASE,
)
MEGACAP_RE = re.compile(
    r"\b(?:apple|microsoft|nvidia|amazon|alphabet|google|meta|tesla|"
    r"berkshire\s+hathaway|jpmorgan|exxon)\b",
    re.IGNORECASE,
)
COMPANY_IMPACT_RE = re.compile(
    r"\b(?:stock|shares?|earnings|profit|revenue|sales|guidance|antitrust|"
    r"merger|acquisition|bankruptcy)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NYT Archive months and append only broad-market-relevant "
            "headlines to the S&P 500 headline CSV."
        )
    )
    parser.add_argument(
        "--headlines-csv",
        type=Path,
        default=DEFAULT_HEADLINES_CSV,
        help="Existing headline CSV to extend in place.",
    )
    parser.add_argument(
        "--prices-csv",
        type=Path,
        default=DEFAULT_PRICES_CSV,
        help="S&P 500 price CSV used to assign the CP column.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="First publication date to consider; defaults to one day after the existing CSV.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
        help="Last publication date to consider (default: 2025-12-31).",
    )
    parser.add_argument(
        "--extra-term",
        action="append",
        default=[],
        help="Additional case-insensitive term that marks an article as relevant; repeatable.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=12.1,
        help="Delay between monthly API calls to respect conservative rate limits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report matching records without modifying the headline CSV.",
    )
    return parser.parse_args()


def read_nyt_key() -> str:
    key = os.environ.get("NYT_KEY", "").strip()
    if key:
        return key

    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "NYT_KEY":
                key = value.strip().strip("'\"")
                if key:
                    return key

    raise RuntimeError("NYT_KEY is not set in the environment or in the project .env file.")


def iter_months(start: date, end: date) -> Iterator[tuple[int, int]]:
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current.year, current.month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def article_text(doc: dict) -> str:
    headline = doc.get("headline") or {}
    keywords = doc.get("keywords") or []
    parts = [
        headline.get("main", ""),
        headline.get("print_headline", ""),
        doc.get("abstract", ""),
        doc.get("snippet", ""),
        doc.get("lead_paragraph", ""),
        doc.get("section_name", ""),
        doc.get("subsection_name", ""),
        doc.get("news_desk", ""),
        " ".join(str(keyword.get("value", "")) for keyword in keywords),
    ]
    return " ".join(str(part) for part in parts if part)


def relevance_reason(doc: dict, extra_terms: Iterable[str] = ()) -> str | None:
    text = article_text(doc)
    if DIRECT_MARKET_RE.search(text):
        return "broad_market"
    if MACRO_RE.search(text) and MARKET_IMPACT_RE.search(text):
        return "macro_market"
    if MEGACAP_RE.search(text) and COMPANY_IMPACT_RE.search(text):
        return "index_constituent"
    if any(term.casefold() in text.casefold() for term in extra_terms if term.strip()):
        return "extra_term"
    return None


def publication_date(doc: dict) -> date | None:
    pub_date = doc.get("pub_date")
    if not pub_date:
        return None
    try:
        return pd.Timestamp(pub_date).date()
    except (TypeError, ValueError):
        return None


def fetch_archive_month(
    session: requests.Session, api_key: str, year: int, month: int
) -> list[dict]:
    url = ARCHIVE_URL.format(year=year, month=month)
    for attempt in range(4):
        response = session.get(url, params={"api-key": api_key}, timeout=120)
        if response.status_code == 429 and attempt < 3:
            wait_seconds = 30 * (attempt + 1)
            print(f"  Rate limited for {year}-{month:02d}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", {}).get("docs", [])
    return []


def load_existing_headlines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Title", "Date", "CP"])
    headlines = pd.read_csv(path, parse_dates=["Date"])
    required = {"Title", "Date", "CP"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    headlines["Date"] = (
        pd.to_datetime(headlines["Date"]).astype("datetime64[ns]").dt.normalize()
    )
    return headlines[["Title", "Date", "CP"]]


def assign_close_prices(records: list[dict], prices_path: Path) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Title", "Date", "CP"])
    prices = pd.read_csv(prices_path, parse_dates=["Date"], usecols=["Date", "Close"])
    prices["Date"] = prices["Date"].astype("datetime64[ns]").dt.normalize()
    prices = prices.sort_values("Date").rename(columns={"Close": "CP"})

    selected = pd.DataFrame(records)
    selected["Date"] = (
        pd.to_datetime(selected["Date"]).astype("datetime64[ns]").dt.normalize()
    )
    selected = selected.sort_values("Date")
    selected = pd.merge_asof(selected, prices, on="Date", direction="backward")
    if selected["CP"].isna().any():
        bad_dates = selected.loc[selected["CP"].isna(), "Date"].dt.date.unique()
        raise ValueError(f"No historical close is available for headline dates: {bad_dates}")
    return selected[["Title", "Date", "CP"]]


def write_updated_headlines(existing: pd.DataFrame, new_rows: pd.DataFrame, path: Path) -> int:
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values(["Date", "Title"], kind="stable").reset_index(drop=True)
    combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    combined.to_csv(temp_path, index=False)
    temp_path.replace(path)
    return len(new_rows)


def main() -> None:
    args = parse_args()
    existing = load_existing_headlines(args.headlines_csv)
    if args.start_date:
        start_date = args.start_date
    elif existing.empty:
        raise ValueError("--start-date is required when the headline CSV has no existing rows.")
    else:
        start_date = (existing["Date"].max() + pd.Timedelta(days=1)).date()

    if start_date > args.end_date:
        print(f"Headline data already reaches the requested end date ({args.end_date}).")
        return

    api_key = read_nyt_key()
    months = list(iter_months(start_date, args.end_date))
    print(
        f"Fetching {len(months)} NYT Archive month(s) for "
        f"{start_date} through {args.end_date}."
    )
    print("The Archive API transfers full months; only relevant articles will be stored.")

    selected_records: list[dict] = []
    fetched_articles = 0
    with requests.Session() as session:
        for index, (year, month) in enumerate(months):
            docs = fetch_archive_month(session, api_key, year, month)
            fetched_articles += len(docs)
            monthly_matches = 0
            for doc in docs:
                article_day = publication_date(doc)
                if article_day is None or not (start_date <= article_day <= args.end_date):
                    continue
                reason = relevance_reason(doc, args.extra_term)
                if not reason:
                    continue
                title = str((doc.get("headline") or {}).get("main", "")).strip()
                if not title:
                    continue
                selected_records.append({"Title": title, "Date": article_day})
                monthly_matches += 1
            print(f"  {year}-{month:02d}: {len(docs):,} articles, {monthly_matches:,} selected")
            if index < len(months) - 1 and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    selected = assign_close_prices(selected_records, args.prices_csv)
    selected = selected.drop_duplicates(subset=["Title", "Date"], keep="first")
    already_present = selected.merge(
        existing[["Title", "Date"]], on=["Title", "Date"], how="inner"
    )
    new_rows = selected.merge(
        existing[["Title", "Date"]], on=["Title", "Date"], how="left", indicator=True
    )
    new_rows = new_rows[new_rows["_merge"] == "left_only"].drop(columns="_merge")

    print(
        f"Selected {len(selected):,} relevant unique headlines from "
        f"{fetched_articles:,} NYT articles; {len(already_present):,} already exist."
    )
    if args.dry_run:
        print(f"Dry run: {len(new_rows):,} new headline rows would be written.")
        return
    if new_rows.empty:
        print("No new relevant headline rows to write.")
        return

    added = write_updated_headlines(existing, new_rows, args.headlines_csv)
    print(f"Wrote {added:,} new headline rows to {args.headlines_csv}.")


if __name__ == "__main__":
    main()
