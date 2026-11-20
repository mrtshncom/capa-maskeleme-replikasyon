"""Makale şekilleri (Şekil 1-4). Kullanım: python sekiller.py <veri_temiz.csv> <is_klasoru> <cikti_klasoru> [1,2,3,4]"""
import sys, os, json, math, itertools, collections, tempfile
import numpy as np, pandas as pd
CSV, IS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEC = sys.argv[4].split(",") if len(sys.argv) > 4 else ["1", "2", "3", "4"]
os.makedirs(OUT, exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.unicode_minus": True})
VIRGUL = FuncFormatter(lambda v, p: (f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.0f}").replace(".", ",").replace("-", "−"))
RENK = {"VA": "#2a78d6", "DP": "#eb6834", "PY": "#1baf7a"}
AD = {"VA": "Veri analitiği", "DP": "Dijital pazarlama", "PY": "Proje yönetimi"}
SERI = {"capa": "#2a78d6", "rastgele": "#eb6834", "belirtec": "#1baf7a"}
L = lambda p: json.load(open(os.path.join(IS, p), encoding="utf-8"))

def kontrol():
    K = dict(duzeyler={}, rastgele={}, belirtec={})
    for p in ["cikti_ek/kontrol_C10_C25_tum.json", "cikti_ek/kontrol_M1_M2.json", "cikti_ek/kontrol_M3_M4.json"]:
        k = L(p)
        for a in K:
            for x, v in k.get(a, {}).items(): K[a].setdefault(x, v)
    return K

def prep():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analiz2.py"), encoding="utf-8").read().split('R["maske"] = {}')[0]
    argv = sys.argv; sys.argv = ["analiz2.py", CSV, tempfile.mkdtemp(), "--hizli"]
    g = {"__name__": "sekil"}; exec(compile(src, "analiz2.py", "exec"), g); sys.argv = argv
    return g

# ---------------- Şekil 1
if "1" in SEC:
    K = kontrol(); M, RS, BT = K["duzeyler"], K["rastgele"], K["belirtec"]
    GM = L("cikti_ek/gomme.json"); GD, GR = GM["duzeyler"], GM["rastgele"]
    lv = ["M0", "M1", "M2", "M3", "M4"]; x = np.arange(len(lv))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    def seri(ax, y, renk, mk, ad, ls="-"):
        ax.plot(x, y, color=renk, lw=2, marker=mk, ms=6, ls=ls, label=ad, markeredgecolor="white", markeredgewidth=1)
    # (a) GAA
    ax = axs[0]
    seri(ax, [M[l]["ic_dis"] for l in lv], SERI["capa"], "o", "Çapa terim maskeleme")
    r = [M["M0"]["ic_dis"]] + [RS[l]["ic_dis"]["ort"] for l in lv[1:]]
    lo = [M["M0"]["ic_dis"]] + [RS[l]["ic_dis"]["alt"] for l in lv[1:]]; hi = [M["M0"]["ic_dis"]] + [RS[l]["ic_dis"]["ust"] for l in lv[1:]]
    ax.fill_between(x, lo, hi, color=SERI["rastgele"], alpha=.2, lw=0)
    seri(ax, r, SERI["rastgele"], "s", "Eşlenmiş rastgele maskeleme")
    seri(ax, [M["M0"]["ic_dis"]] + [BT[l]["ic_dis_ort"] for l in lv[1:]], SERI["belirtec"], "^", "Rastgele belirteç silme", ls="--")
    ax.set_title("(a) GAA: iç–dış fark", loc="left", fontsize=8.5); ax.set_ylabel("İç–dış fark")
    # (b) gömme
    ax = axs[1]
    seri(ax, [GD[l]["ic_dis"] for l in lv], SERI["capa"], "o", "Çapa terim maskeleme")
    r = [GD["M0"]["ic_dis"]] + [GR[l]["ic_dis_ort"] for l in lv[1:]]
    lo = [GD["M0"]["ic_dis"]] + [GR[l]["ic_dis_alt"] for l in lv[1:]]; hi = [GD["M0"]["ic_dis"]] + [GR[l]["ic_dis_ust"] for l in lv[1:]]
    ax.fill_between(x, lo, hi, color=SERI["rastgele"], alpha=.2, lw=0)
    seri(ax, r, SERI["rastgele"], "s", "Eşlenmiş rastgele maskeleme")
    ax.set_title("(b) Cümle gömmeleri: iç–dış fark", loc="left", fontsize=8.5)
    # (c) DKB
    ax = axs[2]
    seri(ax, [M[l]["lda_dkb"]["medyan"] for l in lv], SERI["capa"], "o", "LDA, çapa")
    seri(ax, [M["M0"]["lda_dkb"]["medyan"]] + [RS[l]["lda_dkb"]["ort"] for l in lv[1:]], SERI["rastgele"], "s", "LDA, rastgele")
    seri(ax, [M[l]["hdb"]["dkb_tek_grup"] for l in lv], SERI["capa"], "D", "HDBSCAN, çapa", ls=":")
    seri(ax, [M["M0"]["hdb"]["dkb_tek_grup"]] + [RS[l]["hdb_dkb"]["ort"] for l in lv[1:]], SERI["rastgele"], "v", "HDBSCAN, rastgele", ls=":")
    ax.set_title("(c) Yapı–etiket uyumu (DKB)", loc="left", fontsize=8.5); ax.set_ylabel("DKB")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(lv); ax.set_xlabel("Maskeleme düzeyi"); ax.yaxis.set_major_formatter(VIRGUL); ax.set_ylim(bottom=0)
    h1, l1 = axs[0].get_legend_handles_labels(); h3, l3 = axs[2].get_legend_handles_labels()
    fig.legend(h1 + h3, l1 + l3, loc="lower center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil1.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    print("sekil1")

# ---------------- Şekil 2
if "2" in SEC:
    import umap
    g = prep(); df, lab, DUZEY, lsa = g["df"], g["lab"], g["DUZEY"], g["lsa_vectors"]
    GM = L("cikti_ek/gomme.json")
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.2))
    for j, lev in enumerate(["M0", "M3"]):
        t2 = [[t for t in ts if t not in DUZEY[lev]] for ts in df["tokens"]]; keep = np.array([len(t) >= 2 for t in t2])
        _, Zv, _, _ = lsa([" ".join(t) for t, k in zip(t2, keep) if k]); lb = lab[keep]
        U = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42).fit_transform(Zv)
        E = np.array(GM["duzeyler"][lev]["_U2"]); lbE = lab
        for i, (UU, ll, ad) in enumerate([(U, lb, "GAA"), (E, lbE, "Cümle gömmeleri")]):
            ax = axs[i, j]
            for a, mk in [("VA", "o"), ("DP", "s"), ("PY", "^")]:
                m = ll == a; ax.scatter(UU[m, 0], UU[m, 1], s=11, c=RENK[a], marker=mk, alpha=.8, linewidths=0.3, edgecolors="white", label=f"{AD[a]} (n = {m.sum()})")
            ax.set_title(f"({'abcd'[i * 2 + j]}) {ad}, {lev} düzeyi", loc="left", fontsize=8.5)
            ax.set_xlabel("UMAP boyutu 1"); ax.set_ylabel("UMAP boyutu 2"); ax.xaxis.set_major_formatter(VIRGUL); ax.yaxis.set_major_formatter(VIRGUL)
    h = [Line2D([0], [0], marker=mk, ls="", ms=6, markerfacecolor=RENK[a], markeredgecolor="white", label=AD[a]) for a, mk in [("VA", "o"), ("DP", "s"), ("PY", "^")]]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(OUT, "sekil2.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    print("sekil2")

# ---------------- Şekil 3
if "3" in SEC:
    ANA = L("cikti_ana/sonuclar.json"); GE = L("cikti_ek/gecirgenlik_ek.json")
    Y = ANA["gecirgenlik"]["E20"]["gecirgenlik"]; K5 = GE["kat5_E20"]["gecirgenlik"]
    yon = ["VA->DP", "DP->VA", "VA->PY", "PY->VA", "DP->PY", "PY->DP"]
    fig, ax = plt.subplots(figsize=(6.6, 3.0)); y = np.arange(len(yon))
    for d, (S, renk, mk, ad) in enumerate([(Y, SERI["capa"], "o", "200 yarı bölme"), (K5, SERI["rastgele"], "s", "Beş katlı çapraz ölçü (40 tekrar)")]):
        yy = y + (-0.15 if d == 0 else 0.15)
        med = [S[k]["medyan"] for k in yon]; lo = [S[k]["alt"] for k in yon]; hi = [S[k]["ust"] for k in yon]
        ax.hlines(yy, lo, hi, color=renk, lw=2); ax.plot(med, yy, marker=mk, ls="", ms=7, color=renk, markeredgecolor="white", label=ad)
    ax.set_yticks(y); ax.set_yticklabels([k.replace("->", " → ") for k in yon]); ax.invert_yaxis()
    ax.set_xlabel("Göreli geçirgenlik G(a → b), ortanca ve %2,5 ile %97,5 yüzdelikleri"); ax.xaxis.set_major_formatter(VIRGUL); ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil3.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    print("sekil3")

# ---------------- Şekil 4
if "4" in SEC:
    import networkx as nx
    from adjustText import adjust_text
    g = prep(); df, lab, ACIK = g["df"], g["lab"], g["ACIK_CAPA"]
    docsets = [set(t for t in ts if t not in ACIK) for ts in df["tokens"]]; N = len(docsets)
    cnt = collections.Counter(t for s in docsets for t in s)
    nodes = sorted(t for t, c in cnt.items() if c >= 8); ns = set(nodes); co = collections.Counter()
    for s in docsets:
        for a, b in itertools.combinations(sorted(x for x in s if x in ns), 2): co[(a, b)] += 1
    G = nx.Graph(); G.add_nodes_from(nodes)
    for (a, b), c in co.items():
        if c >= 3 and math.log((c * N) / (cnt[a] * cnt[b])) > 0: G.add_edge(a, b, weight=c)
    G.remove_nodes_from(list(nx.isolates(G)))
    cen = pd.read_csv(os.path.join(IS, "cikti_ek/ag_merkezilik_ek.csv")).set_index("terim")
    AG = L("cikti_ek/ag_ek.json")
    assert set(G.nodes) == set(cen.index), "ağ yeniden kurulamadı"
    deg = dict(G.degree()); top = sorted(G.nodes, key=lambda v: (-deg[v], v))[:45]
    sg = G.subgraph(top).copy(); pos = nx.kamada_kawai_layout(sg, weight=None)
    pal = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    nx.draw_networkx_edges(sg, pos, ax=ax, alpha=.13, edge_color="#52514e", width=[0.3 + 0.05 * d["weight"] for _, _, d in sg.edges(data=True)])
    kol = []
    for dengeli in (False, True):
        nl = [v for v in sg.nodes if (cen.H[v] >= 0.90) == dengeli]
        kol.append(nx.draw_networkx_nodes(sg, pos, nodelist=nl, ax=ax, node_size=[60 + 2500 * cen.arasindalik[v] for v in nl],
                               node_color=[pal[int(cen.topluluk[v]) % 5] for v in nl], alpha=.85,
                               edgecolors="#0b0b0b" if dengeli else "#fcfcfb", linewidths=1.4 if dengeli else 0.6))
    texts = [ax.text(pos[v][0], pos[v][1], v, fontsize=7.2, ha="center", va="center", color="#0b0b0b") for v in sg.nodes]
    adjust_text(texts, x=[pos[v][0] for v in sg.nodes], y=[pos[v][1] for v in sg.nodes], ax=ax, expand=(1.3, 1.5), arrowprops=dict(arrowstyle="-", color="#9a9994", lw=0.4))
    tp = AG["topluluklar"]
    nm = {"VA": "veri analitiği", "DP": "dijital pazarlama", "PY": "proje yönetimi"}
    def tadi(i):
        b = tp[i]["baskin_dagilim"]; t_ = max(b, key=b.get); share = b[t_] / sum(b.values())
        return f"T{i + 1}: " + (f"ağırlıkla {nm[t_]}" if share >= 0.6 else "karma") + f" ({tp[i]['n']} terim)"
    h = [Line2D([0], [0], marker="o", ls="", ms=7, markerfacecolor=pal[i], markeredgewidth=0, label=tadi(i)) for i in range(len(tp))]
    h.append(Line2D([0], [0], marker="o", ls="", ms=7, markerfacecolor="white", markeredgecolor="#0b0b0b", markeredgewidth=1.4, label="Dengeli alan dağılımı (H ≥ 0,90)"))
    for s_, l_ in [(0.01, "Arasındalık 0,01"), (0.10, "Arasındalık 0,10")]:
        h.append(Line2D([0], [0], marker="o", ls="", ms=math.sqrt(60 + 2500 * s_), markerfacecolor="#c3c2b7", markeredgewidth=0, label=l_))
    ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelspacing=1.1)
    ax.axis("off"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil4.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    print("sekil4", len(G), G.number_of_edges())
