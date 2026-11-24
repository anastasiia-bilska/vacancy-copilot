from copilot.render import render_hits
from copilot.search import Hit, search
from tests.test_index import fake_embed, vac
from tests.test_search import make


def hit(**kw):
    return Hit(**{**dict(id="dou:1", title="Python Dev", link="https://x/1", source="dou", score=1.0), **kw})


def test_render_has_title_link_and_metadata():
    out = render_hits([hit(salary_min=1000, salary_max=2000, salary_currency="USD", salary_period="month",
                           work_format="remote", locations=["Kyiv"], skills=["python", "sql"],
                           links=["https://jobs.dou.ua/1", "https://djinni.co/2"])], query="Python dev")
    assert 'href="https://x/1"' in out and ">Python Dev</a>" in out
    for s in ("1,000–2,000", "USD / month", "Remote", "Kyiv", ">DOU", ">Djinni", "several sites"):
        assert s in out
    assert '<li class="vc-tag is-match">python</li>' in out and '<li class="vc-tag">sql</li>' in out


def test_render_empty():
    assert "Нічого не знайдено" in render_hits([])


def test_description_never_rendered(tmp_path):
    t = make(tmp_path, [vac(1, "Python Developer", description="SECRET-TEXT")])
    hits = search(t, "Python", mode="bm25", embed_fn=fake_embed)
    assert hits and "SECRET-TEXT" not in render_hits(hits)


def test_title_escaped_and_only_http_links():
    out = render_hits([hit(title='<script>x</script>"', link="javascript:alert(1)", links=["javascript:alert(1)"])])
    assert "<script>" not in out and "javascript:" not in out
    assert "&lt;script&gt;" in out
    out = render_hits([hit(link='https://x/a"onmouseover=1')])
    assert 'href="https://x/a&quot;onmouseover=1"' in out


def test_app_builds_and_bad_salary_is_a_message(monkeypatch, tmp_path):
    import app
    t = make(tmp_path, [vac(1, "Python", locations=["Kyiv", "Київ"]), vac(2, "Go", locations=["Kyiv"])])
    monkeypatch.setattr(app, "_table", lambda: t)
    opts = app.choices(t)
    assert opts["locations"] == ["Kyiv", "Київ"]  # values from the index, most frequent first
    app.build_ui(opts)
    out = app.run_search("python", "bm25", "", "", "", [], "", "", float("inf"))
    assert "Invalid request" in out
    assert "Invalid request" in app.run_search("python", "junk", "", "", "", [], "", "", None)
