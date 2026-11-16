from datetime import datetime, timezone

from copilot.sync import Vacancy, load, report, row_to_vacancy, save

NONE_COLS = dict(role=None, seniority=None, salary_min=None, salary_max=None, salary_currency=None,
                 salary_period=None, work_format=None, locations=None, english=None, domain=None,
                 skills=None)


def row(**kw):
    base = dict(source="dou", external_id="123", raw_item_id=7, title="Dev", link="https://x/1",
                description="d", last_seen=datetime(2026, 11, 1, tzinfo=timezone.utc), **NONE_COLS)
    return {**base, **kw}


def vac(**kw):
    return row_to_vacancy(row(**kw))


def test_row_to_vacancy_builds_id():
    assert vac().id == "dou:123"


def test_row_to_vacancy_without_extraction():
    v = vac()
    assert v.skills == [] and v.locations == [] and v.role is None


def test_save_load_roundtrip(tmp_path):
    vs = [
        vac(),
        vac(source="djinni", external_id="9", role="backend", salary_min=1000.0, salary_max=2000.0,
            salary_currency="USD", salary_period="month", locations=["Kyiv"], skills=["Python", "SQL"]),
    ]
    p = str(tmp_path / "v.parquet")
    save(vs, p)
    assert load(p) == vs


def test_sync_output_has_no_description(capsys):
    report([vac(description="SECRET-TEXT")])
    out = capsys.readouterr().out
    assert "SECRET-TEXT" not in out and "dou" in out
