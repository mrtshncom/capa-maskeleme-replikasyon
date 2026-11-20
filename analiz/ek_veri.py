"""Ek tablolar için yardımcı çıktılar: belge başına sözlük biçimi istatistikleri, durak sözcük ve maskeleme listeleri."""
import sys, os, json, tempfile, statistics
CSV, OUT = sys.argv[1], sys.argv[2]
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analiz2.py"), encoding="utf-8").read().split('R["maske"] = {}')[0]
sys.argv = ["analiz2.py", CSV, tempfile.mkdtemp(), "--hizli"]
g = {"__name__": "ek"}
exec(compile(_src, "analiz2.py", "exec"), g)
df, Z, ACIK, DUZEY, ALANLAR = g["df"], g["Z"], g["ACIK_CAPA"], g["DUZEY"], g["ALANLAR"]
L = [len(t) for t in df["tokens"]]
def iadeli(k):
    s = set()
    for a in ALANLAR:
        s |= set([w for w, _ in sorted(Z[a].items(), key=lambda x: -x[1]) if w not in ACIK][:k])
    return s
R = dict(lemma=dict(ort=statistics.mean(L), ss=statistics.stdev(L), medyan=statistics.median(L), min=min(L), max=max(L), toplam=sum(L),
                    farkli=len({t for ts in df["tokens"] for t in ts})),
         ek_durak=sorted(g["EK_STOP"]),
         kumulatif={k: sorted(v) for k, v in DUZEY.items()},
         iadeli={f"C{k}": sorted(iadeli(k)) for k in (10, 25, 50)},
         ilk50={a: [[w, float(z)] for w, z in sorted(Z[a].items(), key=lambda x: -x[1])[:50]] for a in ALANLAR})
json.dump(R, open(os.path.join(OUT, "ek_veri.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(R["lemma"], len(R["ek_durak"]), {k: len(v) for k, v in R["kumulatif"].items()}, {k: len(v) for k, v in R["iadeli"].items()})
