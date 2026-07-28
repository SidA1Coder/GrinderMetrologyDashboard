"""Product-info lookup from the SQL ODS database.

Given a corner image, we need its context: which line/station it came from,
the part/serial id, and the corner position (1-4). In production this is read
from the SQL ODS. When no connection string is configured the module falls
back to deterministic MOCK data derived from the filename, so the whole
dashboard is demonstrable without the real database.

Expected image filename convention (configurable via parsing below):
    <LINE>_<SERIAL>_C<CORNER>.jpg      e.g.  Line-A_SN12345_C3.jpg
Anything that does not match still works -- fields are filled from the ODS
query (by parsed serial) or mocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config

_FILENAME_RE = re.compile(
    r"^(?P<location>[^_]+)_(?P<serial>[^_]+)_C(?P<corner>\d+)", re.IGNORECASE
)


@dataclass
class ProductInfo:
    location: str
    part_id: str
    corner: int | None
    grinder: str | None = None  # friendly "Grinder A".. lineage label
    grind_time: datetime | None = None  # when the plate was ground


def parse_filename(image_path: str | Path) -> dict:
    """Best-effort extraction of fields directly from the filename."""
    stem = Path(image_path).stem
    m = _FILENAME_RE.match(stem)
    if m:
        return {
            "location": m.group("location"),
            "serial": m.group("serial"),
            "corner": int(m.group("corner")),
        }
    return {"location": None, "serial": stem, "corner": None}


def _mock_info(parsed: dict) -> ProductInfo:
    """Deterministic pseudo product info so demos are reproducible."""
    serial = parsed.get("serial") or "UNKNOWN"
    location = parsed.get("location")
    if not location:
        idx = sum(ord(ch) for ch in serial) % len(config.LOCATIONS)
        location = config.LOCATIONS[idx]
    corner = parsed.get("corner")
    if corner is None:
        corner = (sum(ord(ch) for ch in serial) % 4) + 1
    return ProductInfo(location=location, part_id=serial, corner=corner)


def _ods_info(parsed: dict) -> ProductInfo | None:
    """Query the real SQL ODS. Returns None on any failure (falls back)."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(config.ODS_CONN_STR)
        with engine.connect() as conn:
            row = conn.execute(
                text(config.ODS_QUERY), {"key": parsed.get("serial")}
            ).fetchone()
        if row is None:
            return None
        data = row._mapping
        return ProductInfo(
            location=data.get("location") or parsed.get("location"),
            part_id=data.get("part_id") or parsed.get("serial"),
            corner=data.get("corner") or parsed.get("corner"),
        )
    except Exception:
        # Any driver/connection error -> fall back to mock rather than crash.
        return None


def get_product_info(image_path: str | Path) -> ProductInfo:
    parsed = parse_filename(image_path)
    if not config.use_mock_db():
        info = _ods_info(parsed)
        if info is not None:
            return _attach_grinder(image_path, info)
    return _attach_grinder(image_path, _mock_info(parsed))


def _attach_grinder(image_path: str | Path, info: ProductInfo) -> ProductInfo:
    """Enrich product info with grinder lineage + grind time via the SubID.

    The corner images on the share are named ``<SubID>_L.bmp`` / ``_R.bmp``.
    Stripping that suffix yields the plate SubID, which resolves (through the
    marker/grinder tables) to the grinder that ground it and the grind time --
    exactly the join the edge-grind KPO uses. When the grinder cannot be
    resolved (offline/mock, or no match) the base info is returned unchanged and
    the grinder lineage becomes the panel's location so downstream grouping and
    filtering still work.
    """
    try:
        import metrology

        sub_id = metrology.sub_id_from_image(image_path)
        gdf = metrology.load_grinder_info([sub_id])
        if not gdf.empty:
            row = gdf.iloc[0]
            info.grinder = metrology.grinder_label(str(row["EquipmentName"]))
            gt = row["GrindTime"]
            info.grind_time = None if gt is None or gt != gt else gt.to_pydatetime()
    except Exception:
        pass
    if info.grinder and (not info.location or info.location == "UNKNOWN"):
        info.location = info.grinder
    return info
