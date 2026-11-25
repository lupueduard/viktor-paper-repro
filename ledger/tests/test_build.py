import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build  # noqa: E402

GOOD = """
[paper]
title = "A | piped title"
arxiv = "2601.01234"
folder = "piped"
pr = 7
tested = 2026-12-01
setup = "tiny setup"

[[claims]]
claim = "claim one"
verdict = "did-not-hold"
evidence = "no effect"

[[claims]]
claim = "claim two"
verdict = "held"
evidence = "it held"
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_repo_entries_are_valid_and_ledger_is_current():
    assert build.load_all(), "no paper files found"
    assert build.main(["--check"]) == 0


def test_template_is_valid_once_filled(tmp_path):
    text = (build.PAPERS_DIR / "_TEMPLATE.toml").read_text().replace("pr = 0", "pr = 3")
    assert build.load_entry(write(tmp_path, "x.toml", text))["paper"]["pr"] == 3


def test_render_rows_order_and_escaping(tmp_path):
    write(tmp_path, "piped.toml", GOOD)
    write(tmp_path, "_ignored.toml", "not = [valid")
    older = GOOD.replace("2026-12-01", "2026-01-01").replace("A | piped title", "Older")
    write(tmp_path, "older.toml", older)
    out = build.render(build.load_all(tmp_path))
    rows = [l for l in out.splitlines() if l.startswith("| [")]
    assert "A \\| piped title" in rows[0] and "Older" in rows[1]  # newest first, pipes escaped
    assert "2 papers, 4 claims: 2 held, 0 partly, 2 didn't hold." in out
    assert "[#7](https://github.com/lupueduard/viktor-paper-repro/pull/7)" in out
    assert out.count("| 〃 | claim two | 〃 | ✅ Held | it held | 〃 |") == 2


@pytest.mark.parametrize(
    "old,new,msg",
    [
        ('verdict = "held"', 'verdict = "mostly"', "use one of held, partly, did-not-hold"),
        ('arxiv = "2601.01234"', 'arxiv = "2601.01234v2"', "no version"),
        ("pr = 7", 'pr = "7"', "pull request number"),
        ("tested = 2026-12-01", 'tested = "2026-12-01"', "unquoted date"),
        ('setup = "tiny setup"', 'setup = " "', "empty setup"),
        ('evidence = "it held"', 'evidnce = "it held"', "missing evidence"),
    ],
)
def test_invalid_entries_are_rejected(tmp_path, old, new, msg):
    with pytest.raises(build.LedgerError, match=msg):
        build.load_entry(write(tmp_path, "bad.toml", GOOD.replace(old, new)))


def test_check_flags_stale_ledger(tmp_path, monkeypatch):
    papers = tmp_path / "ledger" / "papers"
    papers.mkdir(parents=True)
    shutil.copy(build.PAPERS_DIR / "winners-curse.toml", papers)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "PAPERS_DIR", papers)
    monkeypatch.setattr(build, "LEDGER", tmp_path / "LEDGER.md")
    assert build.main(["--check"]) == 1
    assert build.main([]) == 0
    assert build.main(["--check"]) == 0
