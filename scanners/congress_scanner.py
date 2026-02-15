"""
Gann Sentinel Trader - Congress Scanner
Tracks House member stock trades from STOCK Act disclosures (Periodic Transaction Reports).
Uses House Clerk public data - no paid APIs. Based on JayCh99/congress-trades approach.

Tracks high-profile members (e.g. Pelosi) as leading indicators.
Disclosure lag: up to 45 days per STOCK Act.

Version: 1.0.0
Last Updated: February 2026
"""

import io
import logging
import os
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import Config
from models.signals import (
    AssetScope,
    DirectionalBias,
    Evidence,
    Signal,
    SignalSource,
    SignalType,
    TimeHorizon,
)

logger = logging.getLogger(__name__)

# Dollar ranges as they appear in PTR forms
TRANSACTION_DOLLAR_RANGES = [
    "$1,001 - $15,000",
    "$15,001 - $50,000",
    "$50,001 - $100,000",
    "$100,001 - $250,000",
    "$250,001 - $500,000",
    "$500,001 - $1,000,000",
    "$1,000,001 - $5,000,000",
    "$5,000,001 - $25,000,000",
    "$25,000,001 - $50,000,000",
    "Over $50,000,000",
    "Transaction in a Spouse or Dependent Child Asset over $1,000,000",
]
EXTRA_TRANSACTION_HEADERS = ["FILING STATUS", "SUBHOLDING OF", "DESCRIPTION", "LOCATION", "F S", "S O"]

# PDF junk that gets mistaken for asset names (headers, footers, form labels)
ASSET_NAME_JUNK_PATTERNS = [
    r"FILING\s+ID\s*#?\d*",
    r"P\s+T\s+R\s*",
    r"CLERK\s+OF\s+THE",
    r"ID\s+OWNER\s+ASSET",
    r"PERIODIC\s+TRANSACTION",
    r"^\d{8,}$",  # doc ID as number
    r"SPINOFF|SURRENDERED|VISTRA",  # partial form text
]


@dataclass
class Filing:
    """House PTR filing metadata from overview XML."""

    last_name: str
    first_name: str
    year: str
    doc_id: str
    filing_date: str = ""

    @property
    def is_online_ptr(self) -> bool:
        """Online PTRs have doc_id starting with '2'."""
        return self.doc_id.startswith("2") if self.doc_id else False

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclass
class CongressTransaction:
    """Parsed transaction from a PTR PDF."""

    asset_name: str
    transaction_type: str  # P=Purchase, S=Sale
    transaction_date: str
    report_date: str
    dollar_range: str
    ticker: str
    member_name: str


def _remove_excess_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


async def _download_overview_zip(client: httpx.AsyncClient, year: str, cache_dir: Path) -> Optional[Path]:
    """Download year overview ZIP if not cached. Returns path to extracted XML or None."""
    xml_path = cache_dir / "overviews" / year / f"{year}FD.xml"
    if xml_path.exists():
        return xml_path
    url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
    try:
        r = await client.get(url, timeout=30.0)
        r.raise_for_status()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content), "r") as zf:
            for name in zf.namelist():
                if name.endswith(".xml"):
                    zf.extract(name, cache_dir / "overviews" / year)
                    extracted = Path(cache_dir / "overviews" / year / name)
                    if extracted != xml_path and extracted.exists():
                        extracted.rename(xml_path)
                    return xml_path
        return None
    except Exception as e:
        logger.warning(f"Congress: failed to download overview for {year}: {e}")
        return None


def _parse_overview_xml(xml_path: Path, watchlist: List[str]) -> List[Filing]:
    """Parse overview XML and return filings for watched members (by last name).

    House Clerk FD.xml uses Member elements. Tag names or indexed children may vary.
    Fallback: OVERVIEW_TAG_ORDER = Prefix, Last, First, Suffix, FilingType, StateDst, Year, FilingDate, DocID
    """
    watchlist_lower = {n.strip().lower() for n in watchlist if n}
    filings: List[Filing] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall(".//Member"):
            last_name = ""
            first_name = ""
            year = ""
            doc_id = ""
            filing_date = ""
            last_el = member.find("Last")
            if last_el is not None and last_el.text:
                last_name = (last_el.text or "").strip()
            else:
                children = list(member)
                if len(children) > 1:
                    last_name = (children[1].text or "").strip()
            if not last_name or last_name.lower() not in watchlist_lower:
                continue
            first_el = member.find("First")
            if first_el is not None and first_el.text:
                first_name = (first_el.text or "").strip()
            else:
                children = list(member)
                if len(children) > 2:
                    first_name = (children[2].text or "").strip()
            year_el = member.find("Year")
            if year_el is not None and year_el.text:
                year = (year_el.text or "").strip()
            else:
                children = list(member)
                if len(children) > 6:
                    year = (children[6].text or "").strip()
            doc_el = member.find("DocID")
            if doc_el is not None and doc_el.text:
                doc_id = (doc_el.text or "").strip()
            else:
                children = list(member)
                if len(children) > 8:
                    doc_id = (children[8].text or "").strip()
            filing_date_el = member.find("FilingDate")
            if filing_date_el is not None and filing_date_el.text:
                filing_date = (filing_date_el.text or "").strip()
            elif len(list(member)) > 7:
                filing_date = (list(member)[7].text or "").strip()
            if not doc_id:
                continue
            filings.append(
                Filing(
                    last_name=last_name,
                    first_name=first_name,
                    year=year,
                    doc_id=doc_id,
                    filing_date=filing_date,
                )
            )
    except Exception as e:
        logger.warning(f"Congress: failed to parse overview {xml_path}: {e}")
    return filings


async def _download_ptr_pdf(client: httpx.AsyncClient, filing: Filing, cache_dir: Path) -> Optional[bytes]:
    """Download PTR PDF. Returns content or None."""
    pdf_path = cache_dir / "ptr" / filing.year / f"{filing.doc_id}.pdf"
    if pdf_path.exists():
        return pdf_path.read_bytes()
    url = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{filing.year}/{filing.doc_id}.pdf"
    try:
        r = await client.get(url, timeout=30.0)
        r.raise_for_status()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(r.content)
        return r.content
    except Exception as e:
        logger.debug(f"Congress: failed to download PTR {filing.doc_id}: {e}")
        return None


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PTR PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("Congress: pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                text += (t or "") + "\n"
            return text
    except Exception as e:
        logger.debug(f"Congress: PDF extract failed: {e}")
        return ""


def _is_ptr_report(text: str) -> bool:
    if not text or len(text.splitlines()) < 2:
        return False
    second_line = text.splitlines()[1].upper()
    return "P" in second_line and "T" in second_line and "R" in second_line


def _remove_ptr_headers(text: str) -> str:
    end_marker = "\n* FOR THE COMPLETE LIST OF ASSET TYPE ABBREVIATIONS, PLEASE VISIT"
    header_end = "\nTRANSACTIONS"
    table_headers = "ID OWNER ASSET TRANSACTION DATE NOTIFICATION AMOUNT CAP.\nTYPE DATE GAINS >\n$200?\n"
    t = text.replace(table_headers, "")
    start = t.find(header_end)
    if start >= 0:
        start += len(header_end)
    else:
        start = 0
    end = t.find(end_marker)
    if end < 0:
        end = len(t)
    return t[start:end]


def _separate_transactions(text: str) -> List[str]:
    """Split raw text into individual transaction blocks."""
    transactions: List[str] = []
    cur = ""
    prev_extra = False
    for line in text.splitlines():
        line = line.strip()
        found_extra = any(h in line for h in EXTRA_TRANSACTION_HEADERS)
        if prev_extra and not found_extra and cur.strip():
            transactions.append(cur.strip())
            cur = ""
        cur += " " + line
        prev_extra = found_extra
    if cur.strip():
        transactions.append(cur.strip())
    return transactions


# Ownership/transaction codes that look like tickers but aren't
FALSE_TICKER_CODES = {"P", "S", "SP", "JT", "DC", "ID", "PR", "PT", "C"}


def _parse_transaction_block(block: str, member_name: str) -> Optional[CongressTransaction]:
    """Parse a single transaction block into CongressTransaction."""
    text = block
    ticker = ""
    m = re.search(r"\(([A-Z]{2,5})\)", text)  # 2-5 chars (excludes single P/S)
    if m:
        cand = m.group(1)
        if cand not in FALSE_TICKER_CODES and len(cand) >= 2:
            ticker = cand
        text = text.replace(m.group(0), "")

    dates_m = re.search(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})", text)
    if not dates_m:
        return None
    transaction_date = dates_m.group(1)
    report_date = dates_m.group(2)
    dates_start, dates_end = dates_m.start(), dates_m.end()

    type_start = max(
        text.rfind(" S (PARTIAL) ", 0, dates_start),
        text.rfind(" S ", 0, dates_start),
        text.rfind(" P ", 0, dates_start),
    )
    transaction_type = text[type_start:dates_start].strip() if type_start >= 0 else ""
    asset_name = text[:type_start].strip() if type_start >= 0 else text[:dates_start].strip()

    dollar_range = ""
    for rng in TRANSACTION_DOLLAR_RANGES:
        if rng in text:
            dollar_range = rng
            break

    return CongressTransaction(
        asset_name=asset_name,
        transaction_type=transaction_type,
        transaction_date=transaction_date,
        report_date=report_date,
        dollar_range=dollar_range,
        ticker=ticker,
        member_name=member_name,
    )


def _is_junk_asset_name(name: str) -> bool:
    """Return True if asset_name looks like PDF junk, not a real security."""
    if not name or len(name) < 3:
        return True
    n = name.upper().strip()
    for pat in ASSET_NAME_JUNK_PATTERNS:
        if re.search(pat, n, re.IGNORECASE):
            return True
    if n.startswith("FILING") or n.startswith("ID #") or "CLERK" in n:
        return True
    return False


def _transaction_to_directional_bias(txn: CongressTransaction) -> DirectionalBias:
    t = (txn.transaction_type or "").upper()
    if "P " in t or t.startswith("P"):
        return DirectionalBias.POSITIVE
    if "S " in t or t.startswith("S"):
        return DirectionalBias.NEGATIVE
    return DirectionalBias.MIXED


def _short_dollar_range(rng: str) -> str:
    """Shorten dollar range for display (e.g. '$1,000,001 - $5,000,000' -> '$1M-$5M')."""
    if not rng:
        return ""
    r = rng.upper().replace("$", "").replace(",", "")
    if "1001" in r and "15000" in r:
        return "$1K-$15K"
    if "15001" in r and "50000" in r:
        return "$15K-$50K"
    if "50001" in r and "100000" in r:
        return "$50K-$100K"
    if "100001" in r and "250000" in r:
        return "$100K-$250K"
    if "250001" in r and "500000" in r:
        return "$250K-$500K"
    if "500001" in r and "1000000" in r:
        return "$500K-$1M"
    if "1000001" in r and "5000000" in r:
        return "$1M-$5M"
    if "5000001" in r and "25000000" in r:
        return "$5M-$25M"
    if "25000001" in r and "50000000" in r:
        return "$25M-$50M"
    if "OVER" in rng.upper() and "50" in r:
        return ">$50M"
    if "SPOUSE" in rng.upper() or "DEPENDENT" in rng.upper():
        return ">$1M (spouse/dependent)"
    return rng[:35]  # fallback truncate


def _transaction_to_signal(txn: CongressTransaction) -> Signal:
    """Convert a CongressTransaction to a GST Signal with human-readable summary."""
    bias = _transaction_to_directional_bias(txn)
    tickers = [txn.ticker] if txn.ticker else []
    action = "bought" if bias == DirectionalBias.POSITIVE else "sold"

    # Use ticker as primary identifier; fall back to asset_name only if it's real
    display_asset = txn.ticker or ""
    if not _is_junk_asset_name(txn.asset_name) and txn.asset_name:
        display_asset = f"{txn.asset_name}" + (f" ({txn.ticker})" if txn.ticker else "")
    elif txn.ticker:
        display_asset = txn.ticker
    else:
        display_asset = "securities"  # fallback when we have neither

    amt = _short_dollar_range(txn.dollar_range)
    amt_str = f" {amt}" if amt else ""

    # Clear, actionable summary: who did what, when, and what it means
    summary = (
        f"{txn.member_name} {action} {display_asset}{amt_str} on {txn.transaction_date}. "
        f"STOCK Act PTR disclosure (required within 45 days)."
    )

    evidence_excerpt = (
        f"PTR = Periodic Transaction Report. {txn.transaction_type} on {txn.transaction_date}. "
        f"Amount: {txn.dollar_range or 'not specified'}. "
        f"Use as leading indicator — congressional trades have preceded market moves."
    )

    return Signal(
        signal_id=str(uuid.uuid4()),
        signal_type=SignalType.POLICY,
        source=SignalSource.CONGRESS,
        asset_scope=AssetScope(tickers=tickers),
        summary=summary,
        evidence=[
            Evidence(
                source="House Clerk PTR (STOCK Act)",
                source_tier="official",
                excerpt=evidence_excerpt,
                timestamp_utc=datetime.now(timezone.utc),
            )
        ],
        confidence=0.65,
        directional_bias=bias,
        time_horizon=TimeHorizon.WEEKS,
        novelty="new",
        staleness_seconds=7 * 86400,
    )


class CongressScanner:
    """
    Scanner for U.S. House member stock trades (STOCK Act disclosures).
    Tracks Pelosi and other high-profile members as leading indicators.
    """

    DEFAULT_WATCHLIST = ["Pelosi", "McCarthy", "McConnell", "Schumer", "Johnson"]

    def __init__(self, watchlist: Optional[List[str]] = None, years: Optional[List[str]] = None):
        """
        Initialize the Congress Scanner.

        Args:
            watchlist: Last names to track (e.g. ["Pelosi"]). Default from CONGRESS_WATCHLIST env or DEFAULT_WATCHLIST.
            years: Years to fetch (e.g. ["2025","2024"]). Default: current + prior.
        """
        env_watch = os.getenv("CONGRESS_WATCHLIST", "")
        if watchlist is not None:
            self.watchlist = watchlist
        elif env_watch:
            self.watchlist = [n.strip() for n in env_watch.split(",") if n.strip()]
        else:
            self.watchlist = self.DEFAULT_WATCHLIST
        now = datetime.now(timezone.utc)
        current_year = str(now.year)
        prior_year = str(now.year - 1)
        self.years = years if years is not None else [current_year, prior_year]
        self.cache_dir = Config.BASE_DIR / "data" / "congress_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_error: Optional[str] = None
        logger.info(f"CongressScanner init - watchlist={self.watchlist}, years={self.years}")

    @property
    def is_configured(self) -> bool:
        """Scanner is always configured (uses free House Clerk data)."""
        return True

    async def scan(self) -> List[Signal]:
        """
        Scan for recent congressional trades from watched members.
        Returns list of Signal objects.
        """
        self.last_error = None
        signals: List[Signal] = []
        seen: set = set()

        async with httpx.AsyncClient() as client:
            for year in self.years:
                xml_path = await _download_overview_zip(client, year, self.cache_dir)
                if not xml_path:
                    continue
                filings = _parse_overview_xml(xml_path, self.watchlist)
                ptr_filings = [f for f in filings if f.is_online_ptr]
                for filing in ptr_filings:
                    pdf_bytes = await _download_ptr_pdf(client, filing, self.cache_dir)
                    if not pdf_bytes:
                        continue
                    text = _extract_text_from_pdf(pdf_bytes)
                    if not text or not _is_ptr_report(text):
                        continue
                    body = _remove_ptr_headers(text.upper())
                    blocks = _separate_transactions(body)
                    for block in blocks:
                        txn = _parse_transaction_block(block, filing.full_name)
                        if not txn or not txn.ticker:
                            continue
                        dedup_key = (txn.member_name, txn.ticker, txn.transaction_date, txn.transaction_type)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)
                        sig = _transaction_to_signal(txn)
                        signals.append(sig)
                        logger.info(f"Congress: {txn.member_name} {txn.transaction_type} {txn.ticker} {txn.transaction_date}")

        logger.info(f"Congress scanner found {len(signals)} trades from watched members")
        return signals
