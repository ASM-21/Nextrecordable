"""Build a synthetic database and model into a directory (CI's container check uses it).

    python scripts/build_synthetic_db.py OUT_DIR [N_ESTABLISHMENTS]

Everything it writes is declared synthetic, so the API and every report say so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nextrecordable import pipeline, provenance  # noqa: E402
from nextrecordable.enforcement import load_inspection_csv, load_violation_csv  # noqa: E402
from nextrecordable.load import load_summary_csv  # noqa: E402
from tests.synthetic import build_synthetic_dataset  # noqa: E402
from tests.synthetic_enforcement import build_synthetic_enforcement  # noqa: E402


def main(out: Path, n: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out / "nextrecordable.duckdb"))
    for year, path in build_synthetic_dataset(out / "raw" / "ita", n_establishments=n).items():
        load_summary_csv(path, year, con)
    insp, viol = build_synthetic_enforcement(out / "raw" / "dol", n_establishments=n)
    load_inspection_csv(con, insp)
    load_violation_csv(con, viol)
    provenance.declare(con, "synthetic", "scripts/build_synthetic_db.py")
    pipeline.build(con)
    pipeline.train(con, model_dir=out / "model", num_boost_round=150)
    con.close()
    print(f"synthetic database and model in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1200)
