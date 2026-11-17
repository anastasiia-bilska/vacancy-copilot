"""Markdown rendering of search hits. Titles, metadata and skills only: never description text."""
import re

from copilot.search import Hit

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|~&:])")


def esc(s: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", s.replace("\n", " "))


def _url(u: str) -> str | None:
    if not re.match(r"https?://", u, re.I):
        return None
    return re.sub(r"[()<>\s\[\]\\]", lambda m: f"%{ord(m.group()):02X}", u)


def _salary(h: Hit) -> str | None:
    lo, hi = h.salary_min, h.salary_max
    if lo is None and hi is None:
        return None
    rng = f"{lo:g}-{hi:g}" if lo is not None and hi is not None and lo != hi else f"{(lo if lo is not None else hi):g}"
    return f"{rng} {h.salary_currency or ''}/{h.salary_period or ''}".strip(" /")


def render_hits(hits: list[Hit]) -> str:
    if not hits:
        return "Нічого не знайдено / Nothing found."
    out = []
    for n, h in enumerate(hits, 1):
        url = _url(h.link)
        title = f"[{esc(h.title)}]({url})" if url else esc(h.title)
        meta = [h.source, _salary(h), h.work_format, ", ".join(h.locations or []) or None]
        line = f"{n}. **{title}**  \n   " + " · ".join(esc(m) for m in meta if m)
        if h.skills:
            line += "  \n   " + ", ".join(esc(s) for s in h.skills[:8])
        out.append(line)
    return "\n\n".join(out)
