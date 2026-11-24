"""HTML rendering of search hits with the UI kit (copilot/ui.css). Titles, metadata and skills
only: never description text. Every value is escaped; links must be http(s)."""
import re
from html import escape
from urllib.parse import urlparse

from copilot.search import Hit

SOURCES = {"dou.ua": "DOU", "djinni.co": "Djinni"}
FORMATS = {"remote": "Remote", "office": "Office", "hybrid": "Hybrid"}
_SVG = '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"{st}>{p}</svg>'
ARROW = _SVG.format(s=11, w=2.8, st="", p='<path d="M7 17 17 7M9 7h8v8"/>')
SCREEN = _SVG.format(s=15, w=2, st=' style="color:var(--muted)"', p='<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>')
PIN = _SVG.format(s=15, w=2, st=' style="color:var(--muted)"', p='<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21Z"/><circle cx="12" cy="9.5" r="2.5"/>')
DASH = _SVG.format(s=15, w=2, st="", p='<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/>')
ALERT = _SVG.format(s=18, w=2, st="", p='<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/>')


def _url(u: str) -> str | None:
    return escape(u, quote=True) if re.match(r"https?://", u or "", re.I) else None


def _source(u: str) -> str:
    host = (urlparse(u).hostname or "").lower()
    return next((n for d, n in SOURCES.items() if host == d or host.endswith("." + d)), host or "link")


def _words(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _salary(h: Hit) -> str:
    lo, hi = h.salary_min, h.salary_max
    if lo is None and hi is None:
        return f'<span style="display:inline-flex;align-items:center;gap:6px;color:var(--muted)">{DASH}Salary not stated</span>'
    rng = f"{lo:,.0f}–{hi:,.0f}" if lo is not None and hi is not None and lo != hi else f"{(lo if lo is not None else hi):,.0f}"
    unit = " / ".join(x for x in (h.salary_currency, h.salary_period) if x)
    return (f'<span style="display:inline-flex;align-items:baseline;gap:6px"><span class="vc-mono" '
            f'style="font-weight:700;color:var(--ink);font-size:15px">{escape(rng)}</span>'
            f'<span style="color:var(--muted);font-size:13px">{escape(unit)}</span></span>')


def card(h: Hit, query: str = "", top: float = 0.0) -> str:
    """One result card. `top` is the best score in the list: the bar shows score relative to it."""
    links = [(u, _url(u)) for u in dict.fromkeys(h.links or [h.link])]
    href = _url(h.link)
    title = f'<a class="vc-title" href="{href}" target="_blank" rel="noopener">{escape(h.title)}</a>' if href else escape(h.title)
    srcs = "".join(f'<a class="vc-src" href="{e}" target="_blank" rel="noopener" aria-label="Open on {escape(_source(u))}">'
                   f'{escape(_source(u))} {ARROW}</a>' for u, e in links if e)
    if len(links) > 1:
        srcs += '<span style="font-size:12.5px;color:var(--muted)">Same job posted on several sites</span>'
    meta = [_salary(h)]
    if h.work_format:
        meta.append(f'<span style="display:inline-flex;align-items:center;gap:6px">{SCREEN}{escape(FORMATS.get(h.work_format, h.work_format))}</span>')
    if h.locations:
        meta.append(f'<span style="display:inline-flex;align-items:center;gap:6px">{PIN}{escape(", ".join(h.locations))}</span>')
    q = _words(query)
    tags = "".join(f'<li class="vc-tag{" is-match" if _words(s) & q else ""}">{escape(s)}</li>' for s in (h.skills or [])[:6])
    # ponytail: relative bar (RRF/BM25 scores aren't 0..1); calibrate if scores get a common scale
    pct = round(100 * h.score / top) if top > 0 else 0
    return f'''<article style="display:flex;flex-direction:column;gap:12px;padding:20px 22px;background:var(--surface);border:1px solid var(--line);border-radius:18px">
<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px">
<h3 style="margin:0;font-size:17px;line-height:1.35;font-weight:700">{title}</h3>
<div title="Relevance" style="flex:none;display:flex;align-items:center;gap:8px;padding-top:4px">
<span aria-hidden="true" style="display:block;width:44px;height:6px;border-radius:3px;background:var(--surface-2);overflow:hidden"><span style="display:block;height:100%;width:{pct}%;background:var(--best-bar)"></span></span>
<span class="vc-mono" style="font-size:12px;color:var(--muted)"><span class="vc-sr">Relevance </span>{h.score:.2f}</span></div></div>
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px">{srcs}</div>
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px 20px;font-size:14px;color:var(--ink-2)">{"".join(meta)}</div>
<ul aria-label="Skills" style="margin:0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:6px">{tags}</ul>
</article>'''


def notice(title: str, text: str = "", kind: str = "info") -> str:
    """Panel for empty / no-results (kind="info") and invalid input (kind="err")."""
    tone = "border-color:var(--err)" if kind == "err" else ""
    icon_bg = "var(--err-soft);color:var(--err)" if kind == "err" else "var(--surface-2);color:var(--muted)"
    return (f'<div class="vc-panel" role="{"alert" if kind == "err" else "status"}" style="padding:18px 20px;display:flex;gap:12px;align-items:flex-start;{tone}">'
            f'<span style="flex:none;display:flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:10px;background:{icon_bg}">{ALERT}</span>'
            f'<div><b style="display:block">{escape(title)}</b><span style="font-size:14px;color:var(--muted)">{escape(text)}</span></div></div>')


def page(body: str) -> str:
    return f'<div class="vc" style="display:flex;flex-direction:column;gap:12px;background:transparent">{body}</div>'


def render_hits(hits: list[Hit], query: str = "") -> str:
    if not hits:
        return page(notice("Нічого не знайдено / Nothing found", "Try other words or fewer filters."))
    top = max(h.score for h in hits)
    return page("".join(card(h, query, top) for h in hits))
