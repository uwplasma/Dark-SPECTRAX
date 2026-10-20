"""Evaluate the antisym gates (antisym_gates.py docstring) from the records -> summary.json."""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ld = lambda n: json.loads((H / f"{n}.json").read_text())  # noqa: E731
fr, fq, ot, cs, ct, cd = (ld(n)["results"] for n in ("frozen", "floquet", "otsi", "cons", "cost", "cond"))
S = {}
for v in ("AW_c0", "AW_c1", "SQ"):
    f = [r for r in fr if r["var"] == v]
    q = [r for r in fq if r["var"] == v]
    S[v] = dict(frozen_max=max(r["dense_max_re"] for r in f), frozen_min=min(r["dense_max_re"] for r in f), n_frozen=len(f),
                n_frozen_gt_1e10=sum(r["dense_max_re"] > 1e-10 for r in f),
                floquet_max=max(r["floquet"] for r in q), n_floquet=len(q), n_floquet_gt_1e10=sum(r["floquet"] > 1e-10 for r in q))
g1 = S["SQ"]["n_frozen_gt_1e10"] == 0 and S["SQ"]["n_floquet_gt_1e10"] == 0
gate = [r for r in ot if abs(r["aw_rel"]) <= 0.01]
g2 = all(abs(r["sq_rel"]) <= 0.01 for r in gate)
td = [r for r in fr if r["var"] == "SQ"]
tdaw = [r for r in fr if r["var"] == "AW_c0"]
sq_c = [r for r in cd if r["var"] == "SQ"]
g5 = (all(r["normality_defect"] <= 1e-12 for r in sq_c) and
      all(abs(r["dense_max_re"] - r["td_rate"]) <= 1e-8 for r in td))
out = dict(G1=g1, G2=g2, G2_points=len(gate), G2_max_rel=max(abs(r["sq_rel"]) for r in gate),
           G2_max_sq_vs_aw=max(abs(r["sq_vs_aw"]) for r in ot), G5=g5, summary=S,
           SQ_td_max_abs=max(abs(r["td_rate"]) for r in td), AW_td_max=max(r["td_rate"] for r in tdaw),
           AW_td_min=min(r["td_rate"] for r in tdaw),
           AW_dense_vs_td_max=max(abs(r["dense_max_re"] - r["td_rate"]) for r in tdaw),
           SQ_dense_vs_td_max=max(abs(r["dense_max_re"] - r["td_rate"]) for r in td),
           G4_ratio=[r["rhs_ratio"] for r in ct], G4_flag=any(r["rhs_ratio"] > 2 for r in ct), cond=cd, cons=cs,
           verdict="SQ passes G1, G2, G5 (linear only)" if (g1 and g2 and g5) else "SQ REJECTED or G5 failed")
(H / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("cond", "cons")}, indent=1))
