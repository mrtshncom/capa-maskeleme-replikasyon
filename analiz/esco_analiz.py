"""Dış doğrulama: ESCO beceri tanımları derleminde çapa terim ve rastgele maskeleme.

Kullanım: python3 esco_analiz.py <esco_ham.json> <qa_veri_temiz.csv> <cikti_klasoru>
Alan etiketi, becerinin bağlı olduğu meslekten gelir (metinden bağımsız). Birden fazla alanın mesleğine bağlı beceriler
ortak beceri sayılır ve etiketli derlemin dışında tutulur. Ön işleme analiz2.py ile aynıdır.
"""
import sys, os, json, collections, tempfile
import numpy as np, pandas as pd
ESCO, QA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "analiz3.py"), QA, tempfile.mkdtemp(), "yok"]
g = {"__name__": "esco", "__file__": sys.argv[0]}
exec(compile(open(sys.argv[0], encoding="utf-8").read(), "analiz3.py", "exec"), g)
d = json.load(open(ESCO, encoding="utf-8"))
alan = collections.defaultdict(set)
for u, m in d["meslek"].items():
    for s in m["temel"] + m["istege"]: alan[s].add(m["alan"])
rows = [dict(uri=u, alan=next(iter(alan[u])), text=b["ad"] + ". " + b["aciklama"]) for u, b in d["beceri"].items() if len(alan[u]) == 1]
E = pd.DataFrame(rows).sort_values("uri").reset_index(drop=True)
R = dict(beceri_toplam=len(d["beceri"]), ortak=sum(1 for u in d["beceri"] if len(alan[u]) > 1),
         ortak_dagilim=collections.Counter("+".join(sorted(alan[u])) for u in d["beceri"] if len(alan[u]) > 1),
         meslekler={m["ad"]: m["alan"] for m in d["meslek"].values()}, cekilme=d["cekilme_tarihi"])
labE = np.asarray(E["alan"].values, dtype=object); R["alan_dagilimi"] = E["alan"].value_counts().to_dict()
TOKE = list(E["text"].map(g["g"]["preprocess"]))
R["kelime_ort"] = float(E["text"].map(lambda s: len(s.split())).mean()); R["lemma_ort"] = float(np.mean([len(t) for t in TOKE]))
ALANLAR = g["ALANLAR"]; ACIK = g["ACIK_CAPA"]
ZE = {a: g["log_odds"](TOKE, labE, a) for a in ALANLAR}
R["logodds_ilk10"] = {a: [(w, round(v, 2)) for w, v in sorted(ZE[a].items(), key=lambda x: -x[1])[:10]] for a in ALANLAR}
def capa(k):
    s = set(ACIK)
    for a in ALANLAR: s |= {w for w, _ in sorted(ZE[a].items(), key=lambda x: -x[1])[:k]}
    return s
LEV = {"M0": set(), "M1": set(ACIK), "M2": capa(10), "M3": capa(25), "M4": capa(50)}
# rastgele havuz ESCO frekanslarına göre
DF = collections.Counter(t for ts in TOKE for t in set(ts)); NT = sum(len(t) for t in TOKE)
g["DFREQ"].clear(); g["DFREQ"].update(DF)
H = sorted([w for w in DF if w not in LEV["M4"]], key=lambda w: DF[w]); g["HAVUZ"][:] = H
g["HAVUZ_DF"] = np.array([DF[w] for w in H]); g["rastgele_esli"].__globals__["HAVUZ_DF"] = g["HAVUZ_DF"]
R["duzey"] = {}; R["rastgele"] = {}
for lev, m in LEV.items():
    o = g["tam_olcu"](m, toks=TOKE, labels=labE, n_seed=10); o.pop("lda_js", None); R["duzey"][lev] = o
    print(lev, o["maske"], o["belge"], round(o["token_kalan"], 3), round(o["ic_dis"], 4), round(o["perm_p"], 4), round(o["siluet"], 4), round(o["nn_disi"], 3), round(o["lda_dkb"]["medyan"], 3), round(o["hdb"]["dkb_tek_grup"], 3), flush=True)
    if lev == "M0": continue
    r = np.random.default_rng(42 + 7000); vals = []
    for i in range(200):
        rm = g["rastgele_esli"](m, r); t2, keep = g["masked"](rm, TOKE); lb = labE[keep]
        _, Zv, _, _ = g["lsa_vectors"]([" ".join(t) for t, k in zip(t2, keep) if k]); v = g["hizli_olcu"](Zv, lb)
        v["token_kalan"] = sum(map(len, t2)) / NT; vals.append(v)
    icd = np.array([v["ic_dis"] for v in vals])
    R["rastgele"][lev] = dict(ic_dis_ort=float(icd.mean()), alt=float(np.percentile(icd, 2.5)), ust=float(np.percentile(icd, 97.5)),
                              p=float((np.sum(icd <= o["ic_dis"]) + 1) / 201), token_kalan=float(np.mean([v["token_kalan"] for v in vals])),
                              siluet_ort=float(np.mean([v["siluet"] for v in vals])), nn_ort=float(np.mean([v["nn_disi"] for v in vals])))
    print("  RASTGELE", json.dumps(R["rastgele"][lev]), flush=True)
p = np.array([np.mean(labE == a) for a in ALANLAR]); R["nn_beklenti"] = float(1 - (p ** 2).sum())
E.to_csv(os.path.join(OUT, "esco_derlem.csv"), index=False)
g["jdump"](R, "esco_sonuclar.json") if False else json.dump(R, open(os.path.join(OUT, "esco_sonuclar.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o)))
print("bitti")
