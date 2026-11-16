from copilot.render import render_hits
from copilot.search import Hit, search
from tests.test_index import fake_embed, vac
from tests.test_search import make


def hit(**kw):
    return Hit(**{**dict(id="dou:1", title="Python Dev", link="https://x/1", source="dou", score=1.0), **kw})


def test_render_has_title_link_and_metadata():
    out = render_hits([hit(salary_min=1000, salary_max=2000, salary_currency="USD", salary_period="month",
                           work_format="remote", locations=["Kyiv"], skills=["python", "sql"])])
    assert "[Python Dev](https://x/1)" in out
    for s in ("dou", "USD", "remote", "Kyiv", "python"):
        assert s in out


def test_render_empty():
    assert "Нічого не знайдено" in render_hits([])


def test_description_never_rendered(tmp_path):
    t = make(tmp_path, [vac(1, "Python Developer", description="SECRET-TEXT")])
    hits = search(t, "Python", mode="bm25", embed_fn=fake_embed)
    assert hits and "SECRET-TEXT" not in render_hits(hits)


def test_title_escaped_and_only_http_links():
    out = render_hits([hit(title="[x](javascript:alert(1)) <b>", link="javascript:alert(1)")])
    assert "](javascript" not in out and "<b>" not in out
    assert "\\[x\\]" in out
    out = render_hits([hit(link="https://x/a)b")])
    assert "https://x/a%29b" in out


def test_app_builds_and_bad_salary_is_a_message(monkeypatch, tmp_path):
    import app
    monkeypatch.setattr(app, "_table", lambda: make(tmp_path, [vac(1, "Python")]))
    app.build_ui()
    out = app.run_search("python", "bm25", "", "", "", "", "", "", float("inf"))
    assert "Invalid filter" in out
