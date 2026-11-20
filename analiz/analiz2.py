"""Revize analiz: temizlenmiş korpus, çapa sözcük maskeleme ve frekans kontrollü geçirgenlik.

Kullanım: python3 analiz2.py <veri_temiz.csv> <cikti_klasoru> [--hizli]
--hizli: LDA tutarlılık taraması yapılmaz (sağlamlık korpusları için).
Tüm rastgele süreçlerde SEED kullanılır.
"""
import json, re, sys, os, math, itertools, collections
import numpy as np, pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import umap, hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
CSV, OUT = sys.argv[1], sys.argv[2]
HIZLI = "--hizli" in sys.argv
os.makedirs(OUT, exist_ok=True)
R = {}
rng = np.random.default_rng(SEED)
ALANLAR = ["VA", "DP", "PY"]

# ---------------------------------------------------------------- 1. Veri ve alan etiketleri
df = pd.read_csv(CSV)
def alan(i):
    if i <= 257: return "VA"
    if i <= 374 or 405 <= i <= 419: return "DP"
    return "PY"
df["alan"] = df["Unnamed: 0"].map(alan)
lab = np.asarray(df["alan"].values, dtype=object)
R["n"] = int(len(df)); R["alan_dagilimi"] = df["alan"].value_counts().to_dict()
df["text"] = df["question"].astype(str) + " " + df["answer"].astype(str)
wc = lambda s: len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", s))
for col in ["question", "answer", "text"]:
    x = df[col].astype(str).map(wc)
    R[f"kelime_{col}"] = dict(ort=float(x.mean()), ss=float(x.std()), medyan=float(x.median()), min=int(x.min()), max=int(x.max()), toplam=int(x.sum()))
R["kelime_belge_alan"] = {a: float(df.loc[lab == a, "text"].map(wc).mean()) for a in ALANLAR}

# ---------------------------------------------------------------- 2. Ön işleme
STOP = set(stopwords.words("english"))
EK_STOP = {"use", "involve", "yes", "provide", "include", "help", "explain", "concept", "also", "way",
           "make", "many", "various", "different", "important", "importance", "example", "like", "base",
           "new", "one", "well", "etc", "may", "could", "would", "need", "allow", "refer", "role",
           "significance", "purpose", "difference", "relationship", "mean", "term", "type", "common",
           "thing", "get", "know", "able", "within", "often", "several", "certain", "specific"}
STOP |= EK_STOP
lem = WordNetLemmatizer()
def wn_pos(t): return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(t[0], wordnet.NOUN)
def preprocess(s):
    toks = re.sub(r"[^a-z\s]", " ", s.lower()).split()
    out = []
    for w, t in nltk.pos_tag(toks):
        if w in STOP or len(w) < 3: continue
        l = lem.lemmatize(w, wn_pos(t))
        if l in STOP or len(l) < 3: continue
        out.append(l)
    return out
df["tokens"] = df["text"].map(preprocess)
R["token_toplam"] = int(df["tokens"].map(len).sum())
R["sozcuk_dagarcigi"] = len(set(t for ts in df["tokens"] for t in ts))

# ---------------------------------------------------------------- 3. Log-odds (bilgilendirici Dirichlet önselli)
def log_odds(token_lists, labels, a):
    """Alan a'nın diğer alanlara karşı z puanları (Monroe vd., 2008)."""
    cA = collections.Counter(t for ts, l in zip(token_lists, labels) if l == a for t in ts)
    cB = collections.Counter(t for ts, l in zip(token_lists, labels) if l != a for t in ts)
    c0 = cA + cB
    nA, nB, n0 = sum(cA.values()), sum(cB.values()), sum(c0.values())
    # önsel: tüm korpus frekansı, toplam önsel ağırlığı = sözcük dağarcığı büyüklüğü
    a0 = len(c0); alpha = {w: a0 * c0[w] / n0 for w in c0}
    z = {}
    for w in c0:
        aw = alpha[w]
        l1 = math.log((cA[w] + aw) / (nA + a0 - cA[w] - aw)); l2 = math.log((cB[w] + aw) / (nB + a0 - cB[w] - aw))
        var = 1 / (cA[w] + aw) + 1 / (cB[w] + aw)
        z[w] = (l1 - l2) / math.sqrt(var)
    return z
Z = {a: log_odds(df["tokens"], lab, a) for a in ALANLAR}
R["logodds_ilk15"] = {a: [(w, round(v, 2)) for w, v in sorted(Z[a].items(), key=lambda x: -x[1])[:15]] for a in ALANLAR}

# ---------------------------------------------------------------- 4. Maskeleme düzeyleri
ACIK_CAPA = {"data", "analytics", "analytic", "analysis", "analyze", "analyst", "marketing", "market", "marketer",
             "digital", "project", "management", "manage", "manager"}
def capa_k(k):
    s = set(ACIK_CAPA)
    for a in ALANLAR:
        s |= {w for w, _ in sorted(Z[a].items(), key=lambda x: -x[1])[:k]}
    return s
DUZEY = {"M0": set(), "M1": ACIK_CAPA, "M2": capa_k(10), "M3": capa_k(25), "M4": capa_k(50)}
R["maske_boyut"] = {k: len(v) for k, v in DUZEY.items()}
R["maske_M2_terimler"] = sorted(DUZEY["M2"])
all_tok = [t for ts in df["tokens"] for t in ts]

def lsa_vectors(docs):
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, norm="l2")
    X = tfv.fit_transform(docs)
    k = min(100, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    return X, normalize(svd.fit_transform(X)), float(svd.explained_variance_ratio_.sum()), tfv

def within_between(S, lb):
    same = lb[:, None] == lb[None, :]; off = ~np.eye(len(lb), dtype=bool)
    return np.nanmean(S[same & off]) - np.nanmean(S[~same])

def run_lda(tokens, k):
    dic = Dictionary(tokens); dic.filter_extremes(no_below=2, no_above=0.5)
    corp = [dic.doc2bow(t) for t in tokens]
    m = LdaModel(corp, id2word=dic, num_topics=k, random_state=SEED, passes=30, iterations=400,
                 alpha="auto", eta="auto", chunksize=len(corp))
    th = np.zeros((len(corp), k))
    for i, b in enumerate(corp):
        for t, p in m.get_document_topics(b, minimum_probability=0): th[i, t] = p
    return m, dic, corp, th

def evaluate(level, mask, scan=False, scan_tohum=False):
    toks = [[t for t in ts if t not in mask] for ts in df["tokens"]]
    keep = np.array([len(t) >= 2 for t in toks])
    toks_k = [t for t, k in zip(toks, keep) if k]; lb = lab[keep]
    out = dict(belge=int(keep.sum()), bos_kalan=int((~keep).sum()),
               token_payi_kalan=float(sum(map(len, toks)) / len(all_tok)))
    X, Zv, ev, _ = lsa_vectors([" ".join(t) for t in toks_k])
    out["lsa_varyans"] = ev
    S = cosine_similarity(Zv)
    np.fill_diagonal(S, np.nan)
    obs = within_between(S, lb)
    perm = [within_between(S, rng.permutation(lb)) for _ in range(1000)]
    out["ic_dis_fark"] = float(obs); out["perm_p"] = float((np.sum(np.array(perm) >= obs) + 1) / 1001)
    out["perm_null_ort"] = float(np.mean(perm))
    out["ciftler"] = {f"{a}-{b}": float(np.nanmean(S[np.ix_(lb == a, lb == b)])) for a, b in itertools.combinations_with_replacement(ALANLAR, 2)}
    out["silhouette"] = float(silhouette_score(Zv, lb, metric="cosine"))
    nn = np.nan_to_num(S, nan=-2).argmax(axis=1)
    out["nn_alan_disi"] = float(np.mean(lb[nn] != lb))
    out["nn_alan_disi_alan"] = {a: float(np.mean(lb[nn][lb == a] != a)) for a in ALANLAR}
    # LDA k=3 (alan sayısı) ve tutarlılık taraması
    m3, dic, corp, th3 = run_lda(toks_k, 3)
    out["lda3_ami"] = float(adjusted_mutual_info_score(lb, th3.argmax(1)))
    if scan_tohum:
        amis = []
        for s_ in range(10):
            mt = LdaModel(corp, id2word=dic, num_topics=3, random_state=s_, passes=30, iterations=400,
                          alpha="auto", eta="auto", chunksize=len(corp))
            tht = np.array([[p for _, p in mt.get_document_topics(b, minimum_probability=0)] for b in corp])
            amis.append(float(adjusted_mutual_info_score(lb, tht.argmax(1))))
        out["lda3_tohum"] = dict(ami=amis, medyan=float(np.median(amis)), min=float(min(amis)), max=float(max(amis)))
    out["lda3_konular"] = {int(t): dict(kelimeler=[w for w, _ in m3.show_topic(t, 10)],
                                       alan=pd.Series(lb[th3.argmax(1) == t]).value_counts().to_dict()) for t in range(3)}
    if scan:
        coh, mods = {}, {}
        for k in range(2, 16):
            m, d2, c2, th = run_lda(toks_k, k) if k != 3 else (m3, dic, corp, th3)
            coh[k] = float(CoherenceModel(model=m, texts=toks_k, dictionary=d2, coherence="c_v", topn=10).get_coherence())
            mods[k] = (m, th)
        kb = max(coh, key=coh.get)
        out["lda_cv"] = coh; out["lda_en_iyi_k"] = int(kb)
        out["lda_ami_k"] = {k: float(adjusted_mutual_info_score(lb, mods[k][1].argmax(1))) for k in mods}
        out["lda_en_iyi_ami"] = float(adjusted_mutual_info_score(lb, mods[kb][1].argmax(1)))
        mb, thb = mods[kb]
        out["lda_en_iyi_konular"] = {int(t): dict(kelimeler=[w for w, _ in mb.show_topic(t, 10)], belge=int((thb.argmax(1) == t).sum()),
                                                 alan=pd.Series(lb[thb.argmax(1) == t]).value_counts().to_dict()) for t in range(kb)}
        out["lda_en_iyi_alan_theta"] = {a: [round(float(x), 4) for x in thb[lb == a].mean(0)] for a in ALANLAR}
    # UMAP + HDBSCAN
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED).fit_transform(Zv)
    cl = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_method="eom").fit(U).labels_
    nz = cl != -1
    out["hdb_kume"] = int(len(set(cl)) - (1 if -1 in cl else 0)); out["hdb_gurultu"] = float(np.mean(~nz))
    out["hdb_ami"] = float(adjusted_mutual_info_score(lb[nz], cl[nz])) if nz.sum() > 0 else None
    out["hdb_ami_gurultu_dahil"] = float(adjusted_mutual_info_score(lb, cl))
    out["hdb_tablo"] = {int(c): {a: int(((cl == c) & (lb == a)).sum()) for a in ALANLAR} for c in sorted(set(cl))}
    cen_v = {a: Zv[lb == a].mean(0) for a in ALANLAR}
    out["merkez_benzerlik"] = {f"{a}-{b}": float(cosine_similarity([cen_v[a]], [cen_v[b]])[0, 0]) for a, b in itertools.combinations(ALANLAR, 2)}
    out["_U2"] = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(Zv) if level in ("M0", "M3") else None
    out["_lb"] = lb
    return out

R["maske"] = {}
for lev, mask in DUZEY.items():
    R["maske"][lev] = evaluate(lev, mask, scan=(not HIZLI and lev in ("M0", "M1", "M3")), scan_tohum=not HIZLI)
    print(lev, {k: v for k, v in R["maske"][lev].items() if k in ("belge", "ic_dis_fark", "perm_p", "silhouette", "nn_alan_disi", "lda3_ami", "hdb_ami", "hdb_kume")}, flush=True)

# ---------------------------------------------------------------- 5. Geçirgenlik: yarı-bölme, eşit boyutlu, frekans bantlı imza terimleri
# Frekans bantları (seçim yarısındaki belge frekansı); eşlemeli kurulumda her alandan her banttan aynı sayıda terim alınır.
BANTLAR = [(3, 3), (4, 4), (5, 6), (7, 9), (10, 14), (15, 22), (23, 10**9)]
Z_ESIK = 1.0

def eslemeli_imza(z_all, dfreq, K):
    """Her alan için, bant başına eşit sayıda terim: bant kotası alanlar arasındaki en küçük aday sayısına göre belirlenir."""
    cand = {a: [[w for w, v in sorted(z_all[a].items(), key=lambda x: -x[1]) if v >= Z_ESIK and lo <= dfreq[w] <= hi]
                for lo, hi in BANTLAR] for a in ALANLAR}
    m = [min(len(cand[a][b]) for a in ALANLAR) for b in range(len(BANTLAR))]
    if sum(m) <= K:
        n = m[:]
    else:
        raw = [K * x / sum(m) for x in m]; n = [int(math.floor(x)) for x in raw]
        for b in sorted(range(len(m)), key=lambda b: -(raw[b] - n[b])):
            if sum(n) >= K: break
            if n[b] < m[b]: n[b] += 1
    return {a: [w for b in range(len(BANTLAR)) for w in cand[a][b][:n[b]]] for a in ALANLAR}

def penetration(K, exclude, n_split=200, df_band=(3, None), eslemeli=False):
    res = {f"{a}->{b}": [] for a in ALANLAR for b in ALANLAR if a != b}
    dfs = {a: [] for a in ALANLAR}; nsig = {a: [] for a in ALANLAR}
    idx = np.arange(len(df))
    toks = [[t for t in ts if t not in exclude] for ts in df["tokens"]]
    for s in range(n_split):
        r2 = np.random.default_rng(SEED + s)
        sel = np.zeros(len(df), bool)
        for a in ALANLAR:
            ia = idx[lab == a]; sel[r2.choice(ia, len(ia) // 2, replace=False)] = True
        tsel = [toks[i] for i in idx[sel]]; lsel = lab[sel]
        dfreq = collections.Counter(t for ts in tsel for t in set(ts))
        sig = {}
        z_all = {a: log_odds(tsel, lsel, a) for a in ALANLAR}
        if eslemeli:
            sig = eslemeli_imza(z_all, dfreq, K)
        else:
            for a in ALANLAR:
                cand = [w for w, v in sorted(z_all[a].items(), key=lambda x: -x[1]) if dfreq[w] >= df_band[0] and (df_band[1] is None or dfreq[w] <= df_band[1])]
                sig[a] = cand[:K]
        for a in ALANLAR:
            dfs[a].append(np.mean([dfreq[w] for w in sig[a]])); nsig[a].append(len(sig[a]))
        meas = idx[~sel]
        rate = {}
        for a in ALANLAR:
            for b in ALANLAR:
                docs = [toks[i] for i in meas if lab[i] == b]
                ntok = sum(map(len, docs))
                cnt = sum(1 for ts in docs for t in ts if t in set(sig[a]))
                rate[(a, b)] = cnt / ntok * 1000 if ntok else 0.0
        for a in ALANLAR:
            for b in ALANLAR:
                if a != b:
                    res[f"{a}->{b}"].append(rate[(a, b)] / rate[(a, a)] if rate[(a, a)] > 0 else np.nan)
    summ = {k: dict(medyan=float(np.nanmedian(v)), alt=float(np.nanpercentile(v, 2.5)), ust=float(np.nanpercentile(v, 97.5))) for k, v in res.items()}
    asim = {}
    for a, b in itertools.combinations(ALANLAR, 2):
        d1 = np.array(res[f"{a}->{b}"]) - np.array(res[f"{b}->{a}"])
        asim[f"{a}->{b} eksi {b}->{a}"] = dict(medyan=float(np.nanmedian(d1)), alt=float(np.nanpercentile(d1, 2.5)),
                                              ust=float(np.nanpercentile(d1, 97.5)), pay_pozitif=float(np.nanmean(d1 > 0)))
    return dict(K=K, eslemeli=eslemeli, gecirgenlik=summ, asimetri=asim, imza_df_ort={a: float(np.mean(v)) for a, v in dfs.items()},
                imza_sayisi_ort={a: float(np.mean(v)) for a, v in nsig.items()}), res

R["gecirgenlik"] = {}
raw_res = {}
for name, K, excl, esl in [("E20", 20, set(), True), ("E10", 10, set(), True), ("E30", 30, set(), True), ("E20_capasiz", 20, ACIK_CAPA, True),
                           ("K20", 20, set(), False), ("K20_capasiz", 20, ACIK_CAPA, False)]:
    R["gecirgenlik"][name], raw_res[name] = penetration(K, excl, eslemeli=esl)
    print(name, json.dumps(R["gecirgenlik"][name]["gecirgenlik"]), flush=True)
# tam korpusta K=20 imza terimleri (betimleme amaçlı)
dfull = collections.Counter(t for ts in df["tokens"] for t in set(ts))
R["imza_terimleri_tam"] = {a: [w for w, v in sorted(Z[a].items(), key=lambda x: -x[1]) if dfull[w] >= 3][:20] for a in ALANLAR}

# ---------------------------------------------------------------- 6. Semantik ağ (açık çapalar maskelenmiş)
toksN = [[t for t in ts if t not in ACIK_CAPA] for ts in df["tokens"]]
docsets = [set(t) for t in toksN]
cnt = collections.Counter(t for s in docsets for t in s)
nodes = sorted(t for t, c in cnt.items() if c >= 8)
co = collections.Counter()
for s in docsets:
    for a, b in itertools.combinations(sorted(x for x in s if x in set(nodes)), 2):
        co[(a, b)] += 1
N = len(docsets)
G = nx.Graph()
for t in nodes: G.add_node(t)
for (a, b), c in co.items():
    if c < 3: continue
    pmi = math.log((c * N) / (cnt[a] * cnt[b]))
    if pmi > 0: G.add_edge(a, b, weight=c, pmi=pmi, dist=1 / pmi)
G.remove_nodes_from(list(nx.isolates(G)))
comps = sorted(nx.connected_components(G), key=len, reverse=True)
H = G.subgraph(comps[0]).copy()
R["ag"] = dict(dugum=G.number_of_nodes(), kenar=G.number_of_edges(), yogunluk=float(nx.density(G)),
               bilesen=len(comps), en_buyuk=len(comps[0]), kumelenme=float(nx.average_clustering(G)))
deg = nx.degree_centrality(G); btw = nx.betweenness_centrality(G, weight="dist")
eig = nx.eigenvector_centrality(H, weight="weight", max_iter=5000)
comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
comm_of = {v: i for i, c in enumerate(comms) for v in c}
R["ag"]["topluluk"] = len(comms); R["ag"]["modularite"] = float(nx.community.modularity(G, comms, weight="weight"))
def participation(v):
    k = G.degree(v, weight="weight")
    if k == 0: return 0.0
    per = collections.Counter()
    for u in G[v]: per[comm_of[u]] += G[v][u]["weight"]
    return 1 - sum((x / k) ** 2 for x in per.values())
alan_n = {a: int((lab == a).sum()) for a in ALANLAR}
rows = []
for v in G.nodes:
    p = np.array([sum(1 for s, l in zip(docsets, lab) if v in s and l == a) / alan_n[a] for a in ALANLAR])
    q = p / p.sum(); h = float(-(q[q > 0] * np.log(q[q > 0])).sum() / math.log(3))
    rows.append(dict(terim=v, belge_frekansi=cnt[v], derece=G.degree(v), derece_merk=deg[v], arasindalik=btw[v],
                     ozvektor=eig.get(v, 0.0), katilim=participation(v), topluluk=comm_of[v], VA=p[0], DP=p[1], PY=p[2], alan_entropi=h))
cen = pd.DataFrame(rows).sort_values("arasindalik", ascending=False)
# frekans kontrolü: arasındalığın log(belge frekansı) üzerine regresyon artığı
b1, b0 = np.polyfit(np.log(cen.belge_frekansi), cen.arasindalik, 1)
cen["arasindalik_artik"] = cen.arasindalik - (b0 + b1 * np.log(cen.belge_frekansi))
cen.to_csv(os.path.join(OUT, "ag_merkezilik.csv"), index=False)
q75 = cen.arasindalik_artik.quantile(0.75)
br = cen[(cen.arasindalik_artik >= q75) & (cen.alan_entropi >= 0.90)].sort_values("arasindalik_artik", ascending=False)
R["ag"]["kopru_esik_artik_q75"] = float(q75)
R["ag"]["kopru"] = br[["terim", "belge_frekansi", "arasindalik", "arasindalik_artik", "katilim", "alan_entropi", "VA", "DP", "PY"]].round(4).values.tolist()
R["ag"]["ilk_arasindalik"] = cen.head(12)[["terim", "belge_frekansi", "derece_merk", "arasindalik", "arasindalik_artik", "katilim", "alan_entropi"]].round(4).values.tolist()
R["ag"]["ilk_derece"] = cen.sort_values("derece_merk", ascending=False).head(10)[["terim", "derece_merk"]].round(4).values.tolist()
R["ag"]["ilk_ozvektor"] = cen.sort_values("ozvektor", ascending=False).head(10)[["terim", "ozvektor"]].round(4).values.tolist()
R["ag"]["topluluklar"] = {i: dict(n=len(c), terimler=cen[cen.topluluk == i].sort_values("derece_merk", ascending=False).terim.head(8).tolist(),
                                 alan_profil=cen[cen.topluluk == i][["VA", "DP", "PY"]].mean().round(4).to_dict()) for i, c in enumerate(comms)}
R["ag"]["spearman"] = {"arasindalik_df": float(cen[["arasindalik", "belge_frekansi"]].corr("spearman").iloc[0, 1]),
                       "artik_entropi": float(cen[["arasindalik_artik", "alan_entropi"]].corr("spearman").iloc[0, 1]),
                       "katilim_entropi": float(cen[["katilim", "alan_entropi"]].corr("spearman").iloc[0, 1])}

# ---------------------------------------------------------------- 7. Şekiller
if not HIZLI:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    colr = {"VA": "#1f4e79", "DP": "#c55a11", "PY": "#548235"}; ad = {"VA": "Veri analitiği", "DP": "Dijital pazarlama", "PY": "Proje yönetimi"}
    # Şekil: maskeleme duyarlılığı
    levs = list(DUZEY)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
    ax[0].plot(levs, [R["maske"][l]["ic_dis_fark"] for l in levs], marker="o", color="#333333", label="Gözlenen")
    ax[0].plot(levs, [R["maske"][l]["perm_null_ort"] for l in levs], ls="--", color="#999999", label="Permütasyon ortalaması")
    ax[0].set_ylabel("Alan içi − alanlar arası benzerlik"); ax[0].legend(frameon=False, fontsize=7); ax[0].grid(alpha=.3)
    med = [R["maske"][l]["lda3_tohum"]["medyan"] for l in levs]
    ax[1].plot(levs, med, marker="o", color="#1f4e79", label="LDA (k = 3), 10 tohum ortancası")
    ax[1].fill_between(range(len(levs)), [R["maske"][l]["lda3_tohum"]["min"] for l in levs],
                       [R["maske"][l]["lda3_tohum"]["max"] for l in levs], color="#1f4e79", alpha=.15, lw=0)
    ax[1].plot(levs, [R["maske"][l]["hdb_ami_gurultu_dahil"] for l in levs], marker="s", color="#c55a11", label="UMAP + HDBSCAN")
    ax[1].set_ylabel("DKB (alan etiketleriyle)"); ax[1].set_ylim(-0.05, 1); ax[1].legend(frameon=False, fontsize=7); ax[1].grid(alpha=.3)
    for a_ in ax: a_.set_xlabel("Maskeleme düzeyi")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil_maskeleme.png"), dpi=300); plt.close(fig)
    # Şekil: geçirgenlik
    g = R["gecirgenlik"]["E20"]["gecirgenlik"]; keys = list(g)
    fig, ax = plt.subplots(figsize=(6.2, 3))
    y = np.arange(len(keys))
    med = [g[k]["medyan"] for k in keys]
    ax.errorbar(med, y, xerr=[[g[k]["medyan"] - g[k]["alt"] for k in keys], [g[k]["ust"] - g[k]["medyan"] for k in keys]], fmt="o", color="#333333", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([k.replace("->", " → ") for k in keys]); ax.invert_yaxis()
    ax.set_xlabel("Göreli geçirgenlik (hedef alandaki yoğunluk / kaynak alandaki yoğunluk)"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil_gecirgenlik.png"), dpi=300); plt.close(fig)
    # Şekil: UMAP (M0 ve M3)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for a_, lev, tt in [(axs[0], "M0", "Maskelemesiz (M0)"), (axs[1], "M3", "Çapa terimleri maskelenmiş (M3)")]:
        U2 = R["maske"][lev]["_U2"]; lb = R["maske"][lev]["_lb"]
        for a in ALANLAR:
            m = lb == a; a_.scatter(U2[m, 0], U2[m, 1], s=8, c=colr[a], alpha=.7, label=ad[a], edgecolors="none")
        a_.set_title(tt, fontsize=9); a_.set_xticks([]); a_.set_yticks([])
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil_umap.png"), dpi=300); plt.close(fig)
    # Şekil: ağ
    topn = cen.sort_values("derece_merk", ascending=False).head(60).terim.tolist()
    sg = G.subgraph(topn).copy(); pos = nx.spring_layout(sg, weight="weight", seed=SEED, k=0.9)
    ci = cen.set_index("terim")
    def nc(t):
        r = ci.loc[t]
        if r.alan_entropi >= 0.90: return "#7f7f7f"
        return [colr["VA"], colr["DP"], colr["PY"]][int(np.argmax([r.VA, r.DP, r.PY]))]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    nx.draw_networkx_edges(sg, pos, ax=ax, alpha=.15, width=[0.3 + 0.08 * d["weight"] for _, _, d in sg.edges(data=True)])
    nx.draw_networkx_nodes(sg, pos, ax=ax, node_size=[80 + 6000 * btw[v] for v in sg.nodes], node_color=[nc(v) for v in sg.nodes], alpha=.85, linewidths=0)
    nx.draw_networkx_labels(sg, pos, ax=ax, font_size=7)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", markersize=7, markerfacecolor=c, markeredgewidth=0, label=l) for c, l in
                       [(colr["VA"], "Ağırlıkla veri analitiği"), (colr["DP"], "Ağırlıkla dijital pazarlama"), (colr["PY"], "Ağırlıkla proje yönetimi"), ("#7f7f7f", "Dengeli dağılım (H ≥ 0,90)")]],
              frameon=False, fontsize=7, loc="upper right")
    ax.axis("off"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil_semantik_ag.png"), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- 8. Kayıt
for lev in R["maske"]:
    R["maske"][lev].pop("_U2", None); R["maske"][lev].pop("_lb", None)
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump(R, open(os.path.join(OUT, "sonuclar.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=conv)
json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in raw_res.items()}, open(os.path.join(OUT, "gecirgenlik_bolmeler.json"), "w"))
print("bitti")
