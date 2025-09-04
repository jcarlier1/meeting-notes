from __future__ import annotations

from typing import Dict, List


def merge_segments(existing: List[Dict], new: List[Dict], tolerance_sec: float = 0.75) -> List[Dict]:
    """Merge with simple de-duplication based on timestamp proximity and text equality.
    Keeps order stable, appends only non-duplicate segments.
    """
    out = list(existing)
    for seg in new:
        if not any(_is_dup(seg, e, tolerance_sec) for e in existing):
            out.append(seg)
    return out


def _is_dup(a: Dict, b: Dict, tol: float) -> bool:
    if abs(float(a.get("start", 0)) - float(b.get("start", 0))) <= tol:
        at = a.get("text", "").strip()
        bt = b.get("text", "").strip()
        if at == bt or (at in bt) or (bt in at):
            return True
    return False

