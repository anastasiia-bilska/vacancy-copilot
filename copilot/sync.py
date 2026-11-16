"""Pull the last 30 days of vacancies from IT Pulse (read-only) into a local parquet file."""
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PATH = "data/vacancies.parquet"

# extraction_skill_resolved gives canonical names; unresolved raw names (skill_name NULL) are dropped.
SQL = """
select v.source, v.external_id, v.raw_item_id, v.title, v.link,
       r.payload ->> 'description' as description,
       v.role, v.seniority, v.salary_min, v.salary_max, v.salary_currency, v.salary_period,
       v.work_format, v.locations, v.english, v.domain, v.last_seen,
       (select array_agg(distinct s.skill_name order by s.skill_name)
          from extraction_skill_resolved s
         where s.extraction_id = v.extraction_id and s.skill_name is not null) as skills
from vacancy_current v
join raw_item r on r.id = v.raw_item_id
where v.last_seen >= now() - interval '30 days'
  and v.source in ('dou', 'djinni')
order by v.source, v.external_id
"""


@dataclass
class Vacancy:
    id: str
    source: str
    external_id: str
    raw_item_id: int
    title: str
    link: str
    description: str
    role: str | None
    seniority: str | None
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: str | None
    work_format: str | None
    locations: list[str]
    english: str | None
    domain: str | None
    skills: list[str]
    last_seen: datetime


def row_to_vacancy(row: dict) -> Vacancy:
    num = lambda x: None if x is None else float(x)  # psycopg returns Decimal for numeric
    return Vacancy(
        id=f"{row['source']}:{row['external_id']}",
        source=row["source"],
        external_id=row["external_id"],
        raw_item_id=row["raw_item_id"],
        title=row["title"],
        link=row["link"],
        description=row["description"],
        role=row["role"],
        seniority=row["seniority"],
        salary_min=num(row["salary_min"]),
        salary_max=num(row["salary_max"]),
        salary_currency=row["salary_currency"],
        salary_period=row["salary_period"],
        work_format=row["work_format"],
        locations=list(row["locations"] or []),
        english=row["english"],
        domain=row["domain"],
        skills=list(row["skills"] or []),
        last_seen=row["last_seen"],
    )


def fetch_vacancies(conn) -> list[Vacancy]:
    """conn: psycopg connection opened with row_factory=dict_row."""
    return [row_to_vacancy(r) for r in conn.execute(SQL).fetchall()]


def save(vacancies: list[Vacancy], path: str = PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(v) for v in vacancies]), path)


def load(path: str = PATH) -> list[Vacancy]:
    return [Vacancy(**r) for r in pq.read_table(path).to_pylist()]


def report(vacancies: list[Vacancy]) -> None:
    """Counts per source only; never prints vacancy text."""
    counts: dict[str, int] = {}
    for v in vacancies:
        counts[v.source] = counts.get(v.source, 0) + 1
    for source, n in sorted(counts.items()):
        print(f"{source}: {n}")
    print(f"total: {len(vacancies)}")
