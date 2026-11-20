"""Ek analiz: UMAP + HDBSCAN hiperparametre ızgarası (M0, M1, M3).

Kullanım: python3 ek_izgara.py <veri_temiz.csv> <cikti.json>
Ön işleme, log-odds ve maskeleme analiz2.py ile aynıdır (dosyanın hazırlık bölümü çalıştırılır).
"""
import sys, json, itertools
import numpy as np
CSV, OUT = sys.argv[1], sys.argv[2]
src = open(__file__.replace("ek_izgara.py", "analiz2.py"), encoding="utf-8").read().split('R["maske"] = {}')[0]
import tempfile
sys.argv = ["analiz2.py", CSV, tempfile.mkdtemp(), "--hizli"]
g = {"__name__": "ek_izgara"}
exec(compile(src, "analiz2.py", "exec"), g)
import umap, hdbscan
from sklearn.metrics import adjusted_mutual_info_score
NN, NC, MCS = [5, 10, 15, 30, 50], [2, 5, 10], [5, 10, 20, 40]
res = {}
for lev in ["M0", "M1", "M3"]:
    mask = g["DUZEY"][lev]
    toks = [[t for t in ts if t not in mask] for ts in g["df"]["tokens"]]
    keep = np.array([len(t) >= 2 for t in toks]); lb = g["lab"][keep]
    _, Z, _, _ = g["lsa_vectors"]([" ".join(t) for t, k in zip(toks, keep) if k])
    rows = []
    for nn, nc in itertools.product(NN, NC):
        U = umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine", random_state=42).fit_transform(Z)
        for mcs in MCS:
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs).fit(U).labels_
            rows.append(dict(n_neighbors=nn, n_components=nc, min_cluster_size=mcs, kume=int(len(set(cl)) - (1 if -1 in cl else 0)),
                             gurultu=float(np.mean(cl == -1)), dkb=float(adjusted_mutual_info_score(lb, cl))))
    d = np.array([r["dkb"] for r in rows])
    res[lev] = dict(izgara=rows, ozet=dict(n=len(rows), medyan=float(np.median(d)), q1=float(np.percentile(d, 25)), q3=float(np.percentile(d, 75)),
                                           min=float(d.min()), max=float(d.max())))
    print(lev, json.dumps(res[lev]["ozet"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
