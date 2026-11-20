"""NKB (normalleştirilmiş) ağırlıklı Louvain bölünmesinde karma toplulukların çekirdek terimlerinin birlikteliği."""
import sys, os, json, math, itertools, collections, tempfile
import numpy as np, pandas as pd, networkx as nx
CSV, OUT = sys.argv[1], sys.argv[2]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analiz2.py"), encoding="utf-8").read().split('R["maske"] = {}')[0]
sys.argv = ["analiz2.py", CSV, tempfile.mkdtemp(), "--hizli"]
g = {"__name__": "npmi"}; exec(compile(src, "analiz2.py", "exec"), g)
df, ACIK = g["df"], g["ACIK_CAPA"]
docsets = [set(t for t in ts if t not in ACIK) for ts in df["tokens"]]; N = len(docsets)
cnt = collections.Counter(t for s in docsets for t in s)
nodes = sorted(t for t, c in cnt.items() if c >= 8); ns = set(nodes); co = collections.Counter()
for s in docsets:
    for a, b in itertools.combinations(sorted(x for x in s if x in ns), 2): co[(a, b)] += 1
G = nx.Graph(); G.add_nodes_from(nodes)
for (a, b), c in co.items():
    if c < 3: continue
    pmi = math.log((c * N) / (cnt[a] * cnt[b]))
    if pmi > 0: G.add_edge(a, b, weight=c, npmi=pmi / (-math.log(c / N)))
G.remove_nodes_from(list(nx.isolates(G)))
cm = nx.community.louvain_communities(G, weight="npmi", seed=42); co_n = {v: i for i, c in enumerate(cm) for v in c}
cen = pd.read_csv(os.path.join(OUT, "ag_merkezilik_ek.csv")).set_index("terim")
R = {"npmi_topluluklar": [sorted(c) for c in cm]}
cekirdek = {"T3": ["cluster", "algorithm", "validation", "campaign", "conversion", "track", "effectiveness", "measure", "evaluate", "metric"],
            "T4": ["stakeholder", "plan", "scope", "communication", "method", "dataset", "statistical", "process", "ensure"]}
for k, ts in cekirdek.items():
    ts = [t for t in ts if t in co_n]
    dag = collections.Counter(co_n[t] for t in ts)
    R[k] = dict(terim=len(ts), en_buyuk_ortak=max(dag.values()), dagilim={str(a): b for a, b in dag.items()},
                ana_topluluk={t: int(cen.topluluk[t]) + 1 for t in ts}, npmi_topluluk={t: co_n[t] for t in ts})
    print(k, R[k]["terim"], R[k]["en_buyuk_ortak"], R[k]["dagilim"])
json.dump(R, open(os.path.join(OUT, "npmi_denetim.json"), "w"), ensure_ascii=False, indent=1)
