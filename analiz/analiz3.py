"""Ek analizler: kontrol deneyleri, duyarlılık, gömme, geçirgenlik, ağ ve ızgara.

Kullanım: python3 analiz3.py <veri_temiz.csv> <cikti_klasoru> <parca> [--tekrar N]
parca:
  kontrol   : iadeli ve kümülatif maskeleme, frekansı eşlenmiş rastgele maskeleme, rastgele belirteç silme
  duyarlik  : gürültü işleme, sabit örneklem, GAA boyutu, LDA Jensen-Shannon, VA alt konu denetimi
  gomme     : bağlamsal cümle gömmeleri (all-mpnet-base-v2) ile M0-M4, iadeli düzeyler ve rastgele denetim
  gecirgen  : sayım düzeyleri, alan içi oranla eşleme, tekrarlı 5 katlı çapraz geçirgenlik
  ag        : ağ eşik duyarlılığı, alanlar arası arasındalık, topluluk profilleri, Şekil 4
  izgara    : 60 ayarlık UMAP-HDBSCAN ızgarası, üç gürültü işleme kuralıyla (M0, M1, M3)
Ön işleme, log-odds ve maskeleme düzeyleri analiz2.py ile aynıdır (dosyanın hazırlık bölümü çalıştırılır).
"""
import sys, os, json, math, itertools, collections, tempfile, time
import numpy as np, pandas as pd

CSV, OUT, PARCA = sys.argv[1], sys.argv[2], sys.argv[3]
TEKRAR = int(sys.argv[sys.argv.index("--tekrar") + 1]) if "--tekrar" in sys.argv else 200
os.makedirs(OUT, exist_ok=True)
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analiz2.py"), encoding="utf-8").read().split('R["maske"] = {}')[0]
_argv = sys.argv
sys.argv = ["analiz2.py", CSV, tempfile.mkdtemp(), "--hizli"]
g = {"__name__": "analiz3"}
exec(compile(_src, "analiz2.py", "exec"), g)
sys.argv = _argv
df, lab, Z, ACIK_CAPA, DUZEY, ALANLAR = g["df"], g["lab"], g["Z"], g["ACIK_CAPA"], g["DUZEY"], g["ALANLAR"]
lsa_vectors, within_between, log_odds = g["lsa_vectors"], g["within_between"], g["log_odds"]
SEED = 42
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import umap, hdbscan
TOK = list(df["tokens"])
DFREQ = collections.Counter(t for ts in TOK for t in set(ts))
TOKCNT = collections.Counter(t for ts in TOK for t in ts)
NTOK = sum(TOKCNT.values())

def capa_cekirdek(k):
    """İadeli düzey: açık alan sözcükleri metne geri verilir, her alanın açık alan sözcüğü olmayan en ayırt edici k terimi maskelenir."""
    s = set()
    for a in ALANLAR:
        s |= set([w for w, _ in sorted(Z[a].items(), key=lambda x: -x[1]) if w not in ACIK_CAPA][:k])
    return s

IADELI = {"C10": capa_cekirdek(10), "C25": capa_cekirdek(25), "C50": capa_cekirdek(50)}
TUM_CAPA = set(DUZEY["M4"]) | IADELI["C50"]

def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, set): return sorted(o)
        return str(o)
    json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=conv)

def masked(mask, toks=None):
    toks = TOK if toks is None else toks
    t2 = [[t for t in ts if t not in mask] for ts in toks]
    keep = np.array([len(t) >= 2 for t in t2])
    return t2, keep

def perm_stats(S, lb, n=1000, seed=SEED):
    r = np.random.default_rng(seed)
    obs = within_between(S, lb)
    perm = np.array([within_between(S, r.permutation(lb)) for _ in range(n)])
    return float(obs), float((np.sum(perm >= obs) + 1) / (n + 1)), float((obs - perm.mean()) / perm.std(ddof=1)), float(perm.mean())

def hizli_olcu(Zv, lb, perm=0):
    S = cosine_similarity(Zv); np.fill_diagonal(S, np.nan)
    o = dict(ic_dis=float(within_between(S, lb)), siluet=float(silhouette_score(Zv, lb, metric="cosine")))
    nn = np.nan_to_num(S, nan=-2).argmax(1); o["nn_disi"] = float(np.mean(lb[nn] != lb))
    if perm:
        _, p, zz, mu = perm_stats(S, lb, perm); o.update(perm_p=p, perm_z=zz, perm_ort=mu)
    return o

def lda_theta(toks, k, seed):
    dic = Dictionary(toks); dic.filter_extremes(no_below=2, no_above=0.5)
    corp = [dic.doc2bow(t) for t in toks]
    m = LdaModel(corp, id2word=dic, num_topics=k, random_state=seed, passes=30, iterations=400, alpha="auto", eta="auto", chunksize=len(corp))
    th = np.zeros((len(corp), k))
    for i, b in enumerate(corp):
        for t, p in m.get_document_topics(b, minimum_probability=0): th[i, t] = p
    return th

def js_fark(th, lb):
    """LDA konu dağılımlarında alanlar arası ortalama Jensen-Shannon uzaklığı eksi alan içi ortalama (uzaklık, log2)."""
    from scipy.spatial.distance import jensenshannon
    P = th / th.sum(1, keepdims=True)
    n = len(P); D = np.zeros((n, n))
    for i in range(n): D[i] = jensenshannon(P[i][None, :].repeat(n, 0).T, P.T, base=2)
    D = np.nan_to_num(D, nan=0.0)  # özdeş dağılımlarda kayan nokta hatasıyla oluşan NaN uzaklık 0'dır
    same = lb[:, None] == lb[None, :]; off = ~np.eye(n, dtype=bool)
    return float(D[~same].mean() - D[same & off].mean())

def hdb_uc(cl, lb):
    """Gürültü (-1) için üç kural: tek grup, dışarıda bırakma (kapsama ile), her gürültü belgesi tekil küme."""
    nz = cl != -1
    tek = float(adjusted_mutual_info_score(lb, cl))
    haric = float(adjusted_mutual_info_score(lb[nz], cl[nz])) if nz.sum() > 1 and len(set(cl[nz])) > 1 else 0.0
    c2 = cl.copy(); m = cl.max() + 1
    for i in np.where(~nz)[0]: c2[i] = m; m += 1
    tekil = float(adjusted_mutual_info_score(lb, c2))
    return dict(kume=int(len(set(cl[nz]))), gurultu=float(np.mean(~nz)), dkb_tek_grup=tek, dkb_haric=haric, kapsama=float(nz.mean()), dkb_tekil=tekil)

def umap_hdb(Zv, lb, nn=15, nc=5, mcs=10, seed=SEED):
    U = umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine", random_state=seed).fit_transform(Zv)
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs).fit(U).labels_
    return hdb_uc(cl, lb), cl

def tam_olcu(mask, toks=None, labels=None, n_seed=10, perm=1000, kume=True):
    labels = lab if labels is None else labels
    t2, keep = masked(mask, toks)
    tk = [t for t, k in zip(t2, keep) if k]; lb = labels[keep]
    base = TOK if toks is None else toks
    X, Zv, ev, tfv = lsa_vectors([" ".join(t) for t in tk])
    o = dict(maske=len(mask), belge=int(keep.sum()), cikan=int((~keep).sum()),
             cikan_alan={a: int(((~keep) & (labels == a)).sum()) for a in ALANLAR},
             token_kalan=float(sum(map(len, t2)) / sum(map(len, base))), ozellik=int(X.shape[1]), lsa_varyans=ev,
             lemma_belge_ort=float(np.mean([len(t) for t in tk])))
    o.update(hizli_olcu(Zv, lb, perm))
    if n_seed:
        amis, jss = [], []
        for s in range(n_seed):
            th = lda_theta(tk, 3, s); amis.append(float(adjusted_mutual_info_score(lb, th.argmax(1)))); jss.append(js_fark(th, lb))
        o["lda_dkb"] = dict(medyan=float(np.median(amis)), min=float(min(amis)), max=float(max(amis)), hepsi=amis)
        o["lda_js"] = dict(medyan=float(np.median(jss)), min=float(min(jss)), max=float(max(jss)))
    if kume:
        o["hdb"], _ = umap_hdb(Zv, lb)
    return o

# ------------------------------------------------------------------ rastgele maskeleme: belge frekansı eşlenmiş
HAVUZ = sorted([w for w in DFREQ if w not in TUM_CAPA], key=lambda w: DFREQ[w])
HAVUZ_DF = np.array([DFREQ[w] for w in HAVUZ])

def rastgele_esli(mask, r):
    """Maskelenen her çapa terim için, çapa olmayan havuzdan belge frekansı en yakın 6 terim arasından rastgele bir terim seçilir (iadesiz)."""
    kullan = np.zeros(len(HAVUZ), bool); out = set()
    for w in sorted(mask, key=lambda w: -DFREQ.get(w, 0)):
        d = DFREQ.get(w, 0)
        if d == 0: continue
        uz = np.abs(HAVUZ_DF - d).astype(float); uz[kullan] = np.inf
        aday = np.argsort(uz, kind="stable")[:6]
        aday = aday[np.isfinite(uz[aday])]
        j = r.choice(aday); kullan[j] = True; out.add(HAVUZ[j])
    return out

def rastgele_belirtec(pay, r):
    """Belirteçlerin pay kadarını türden bağımsız olarak rastgele siler (salt seyreklik denetimi)."""
    return [[t for t in ts if r.random() >= pay] for ts in TOK]

SECIM = sys.argv[sys.argv.index("--duzey") + 1].split(",") if "--duzey" in sys.argv else ["M1", "M2", "M3", "M4", "C10", "C25"]
if PARCA == "kontrol":
    R = dict(tekrar=TEKRAR)
    R["duzeyler"] = {}
    for ad, m in [("M0", set()), ("M1", DUZEY["M1"]), ("M2", DUZEY["M2"]), ("M3", DUZEY["M3"]), ("M4", DUZEY["M4"]),
                  ("C10", IADELI["C10"]), ("C25", IADELI["C25"]), ("C50", IADELI["C50"])]:
        if ad != "M0" and ad not in SECIM and "--tum" not in sys.argv: continue
        R["duzeyler"][ad] = tam_olcu(m)
        print(ad, {k: R["duzeyler"][ad][k] for k in ("maske", "belge", "token_kalan", "ic_dis", "perm_z", "siluet", "nn_disi")},
              R["duzeyler"][ad]["lda_dkb"]["medyan"], R["duzeyler"][ad]["hdb"]["dkb_tek_grup"], flush=True)
    R["iadeli_terimler"] = {k: sorted(v) for k, v in IADELI.items()}
    R["kumulatif_terimler"] = {k: sorted(DUZEY[k]) for k in ("M1", "M2", "M3", "M4")}
    # rastgele denetim
    R["rastgele"] = {}
    for ad in [x for x in ["M1", "M2", "M3", "M4", "C10", "C25"] if x in SECIM]:
        m = DUZEY[ad] if ad in DUZEY else IADELI[ad]
        r = np.random.default_rng(SEED + 1000)
        rows = []
        for i in range(TEKRAR):
            rm = rastgele_esli(m, r)
            t2, keep = masked(rm)
            tk = [t for t, k in zip(t2, keep) if k]; lb = lab[keep]
            _, Zv, _, _ = lsa_vectors([" ".join(t) for t in tk])
            o = hizli_olcu(Zv, lb); o["token_kalan"] = float(sum(map(len, t2)) / NTOK); o["belge"] = int(keep.sum())
            o["df_ort"] = float(np.mean([DFREQ[w] for w in rm]))
            if i < 50:
                th = lda_theta(tk, 3, i % 10); o["lda_dkb"] = float(adjusted_mutual_info_score(lb, th.argmax(1)))
                o["hdb"] = umap_hdb(Zv, lb, seed=SEED + i)[0]
            rows.append(o)
        obs = R["duzeyler"][ad]
        icd = np.array([x["ic_dis"] for x in rows]); sil = np.array([x["siluet"] for x in rows]); nnd = np.array([x["nn_disi"] for x in rows])
        lda = np.array([x["lda_dkb"] for x in rows if "lda_dkb" in x]); hdb = np.array([x["hdb"]["dkb_tek_grup"] for x in rows if "hdb" in x])
        q = lambda v: dict(ort=float(v.mean()), ss=float(v.std(ddof=1)), alt=float(np.percentile(v, 2.5)), ust=float(np.percentile(v, 97.5)))
        R["rastgele"][ad] = dict(
            maske_df_ort=float(np.mean([DFREQ[w] for w in m])), rastgele_df_ort=float(np.mean([x["df_ort"] for x in rows])),
            token_kalan=q(np.array([x["token_kalan"] for x in rows])), ic_dis=q(icd), siluet=q(sil), nn_disi=q(nnd), lda_dkb=q(lda), hdb_dkb=q(hdb),
            p_ic_dis=float((np.sum(icd <= obs["ic_dis"]) + 1) / (len(icd) + 1)),
            p_siluet=float((np.sum(sil <= obs["siluet"]) + 1) / (len(sil) + 1)),
            p_nn=float((np.sum(nnd >= obs["nn_disi"]) + 1) / (len(nnd) + 1)),
            p_lda=float((np.sum(lda <= obs["lda_dkb"]["medyan"]) + 1) / (len(lda) + 1)),
            p_hdb=float((np.sum(hdb <= obs["hdb"]["dkb_tek_grup"]) + 1) / (len(hdb) + 1)),
            azalma_capa=float(1 - obs["ic_dis"] / R["duzeyler"]["M0"]["ic_dis"]), azalma_rastgele=float(1 - icd.mean() / R["duzeyler"]["M0"]["ic_dis"]))
        print("RASTGELE", ad, json.dumps({k: v for k, v in R["rastgele"][ad].items() if k in ("ic_dis", "p_ic_dis", "azalma_capa", "azalma_rastgele", "token_kalan", "lda_dkb", "hdb_dkb")}), flush=True)
    # rastgele belirteç silme (aynı belirteç payı)
    R["belirtec"] = {}
    for ad in [x for x in ["M1", "M2", "M3", "M4"] if x in SECIM and "--belirtec" in sys.argv]:
        pay = 1 - R["duzeyler"][ad]["token_kalan"]; r = np.random.default_rng(SEED + 2000); rows = []
        for i in range(min(TEKRAR, 100)):
            t2 = rastgele_belirtec(pay, r); keep = np.array([len(t) >= 2 for t in t2])
            tk = [t for t, k in zip(t2, keep) if k]; lb = lab[keep]
            _, Zv, _, _ = lsa_vectors([" ".join(t) for t in tk]); o = hizli_olcu(Zv, lb); o["belge"] = int(keep.sum()); rows.append(o)
        icd = np.array([x["ic_dis"] for x in rows])
        R["belirtec"][ad] = dict(pay=pay, ic_dis_ort=float(icd.mean()), ic_dis_alt=float(np.percentile(icd, 2.5)), ic_dis_ust=float(np.percentile(icd, 97.5)),
                                 siluet_ort=float(np.mean([x["siluet"] for x in rows])), nn_ort=float(np.mean([x["nn_disi"] for x in rows])),
                                 belge_ort=float(np.mean([x["belge"] for x in rows])))
        print("BELIRTEC", ad, json.dumps(R["belirtec"][ad]), flush=True)
    jdump(R, "kontrol_" + "_".join(SECIM) + ("_tum" if "--tum" in sys.argv else "") + ".json")

if PARCA == "js":
    # LDA konu dağılımlarında Jensen-Shannon farkı (tam_olcu ile aynı tohumlar ve aynı düzeyler)
    R = {}
    for ad in SECIM:
        m = set() if ad == "M0" else (DUZEY[ad] if ad in DUZEY else IADELI[ad])
        t2, keep = masked(m); tk = [t for t, k in zip(t2, keep) if k]; lb = lab[keep]
        jss, amis = [], []
        for s in range(10):
            th = lda_theta(tk, 3, s); jss.append(js_fark(th, lb)); amis.append(float(adjusted_mutual_info_score(lb, th.argmax(1))))
        R[ad] = dict(medyan=float(np.median(jss)), min=float(min(jss)), max=float(max(jss)), hepsi=jss, dkb_denetim=float(np.median(amis)))
        print("JS", ad, R[ad]["medyan"], R[ad]["dkb_denetim"], flush=True)
    jdump(R, "js_" + "_".join(SECIM) + ".json")

if PARCA == "duyarlik":
    R = {}
    # sabit örneklem: M4'te kalan belgeler tüm düzeylerde
    _, keep4 = masked(DUZEY["M4"])
    toks_f = [t for t, k in zip(TOK, keep4) if k]; lab_f = lab[keep4]
    R["m4_cikan_alan"] = {a: int(((~keep4) & (lab == a)).sum()) for a in ALANLAR}
    R["sabit"] = {}
    for ad in ["M0", "M1", "M2", "M3", "M4"]:
        R["sabit"][ad] = tam_olcu(DUZEY[ad], toks=toks_f, labels=lab_f, n_seed=10)
        print("SABIT", ad, R["sabit"][ad]["belge"], R["sabit"][ad]["ic_dis"], R["sabit"][ad]["lda_dkb"]["medyan"], R["sabit"][ad]["hdb"], flush=True)
    # GAA boyutu ve ham TF-IDF
    R["boyut"] = {}
    for ad in ["M0", "M1", "M2", "M3", "M4"]:
        t2, keep = masked(DUZEY[ad]); tk = [" ".join(t) for t, k in zip(t2, keep) if k]; lb = lab[keep]
        tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, norm="l2"); X = tfv.fit_transform(tk)
        row = dict(ozellik=int(X.shape[1]))
        Sx = cosine_similarity(X); np.fill_diagonal(Sx, np.nan); row["tfidf_ic_dis"] = float(within_between(Sx, lb))
        for d in [15, 30, 50, 100]:
            svd = TruncatedSVD(n_components=d, random_state=SEED); Zv = normalize(svd.fit_transform(X))
            o = hizli_olcu(Zv, lb); row[f"d{d}"] = dict(ic_dis=o["ic_dis"], siluet=o["siluet"], nn=o["nn_disi"], varyans=float(svd.explained_variance_ratio_.sum()))
        R["boyut"][ad] = row
        print("BOYUT", ad, json.dumps(row), flush=True)
    # GAA vektörleri üzerinde doğrudan HDBSCAN (UMAP olmadan)
    R["dogrudan_hdb"] = {}
    for ad in ["M0", "M1", "M3"]:
        t2, keep = masked(DUZEY[ad]); tk = [" ".join(t) for t, k in zip(t2, keep) if k]; lb = lab[keep]
        _, Zv, _, _ = lsa_vectors(tk)
        cl = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean").fit(Zv).labels_
        R["dogrudan_hdb"][ad] = hdb_uc(cl, lb)
        print("DOGRUDAN", ad, R["dogrudan_hdb"][ad], flush=True)
    # VA alt konu yapısı: algoritmik (tree, decision, cluster, clustering, algorithm, feature, machine) belgeler
    ALG = {"tree", "decision", "cluster", "clustering", "algorithm", "feature", "machine", "learning", "learn", "classification", "regression", "split", "node", "dimensionality", "centroid", "subset", "pruning"}
    va = lab == "VA"; alg = np.array([len(set(t) & ALG) > 0 for t in TOK])
    R["va_alt"] = dict(va=int(va.sum()), alg=int((va & alg).sum()), tree_decision_cluster=int((va & np.array([len(set(t) & {"tree", "decision", "cluster", "clustering"}) > 0 for t in TOK])).sum()))
    keepg = ~(va & alg)
    toks_g = [t for t, k in zip(TOK, keepg) if k]; lab_g = lab[keepg]
    R["va_genel"] = {}
    for ad in ["M0", "M1", "M2", "M3"]:
        R["va_genel"][ad] = tam_olcu(DUZEY[ad], toks=toks_g, labels=lab_g, n_seed=10)
        print("VAGENEL", ad, R["va_genel"][ad]["belge"], R["va_genel"][ad]["ic_dis"], R["va_genel"][ad]["lda_dkb"]["medyan"], flush=True)
    # dengeli alt örneklem: her alandan 89 belge (50 tekrar)
    R["dengeli"] = {}
    for ad in ["M0", "M1", "M3"]:
        r = np.random.default_rng(SEED + 3000); vals = []
        for i in range(50):
            idx = np.concatenate([r.choice(np.where(lab == a)[0], 89, replace=False) for a in ALANLAR])
            t2, keep = masked(DUZEY[ad], [TOK[j] for j in idx]); lb = lab[idx][keep]
            _, Zv, _, _ = lsa_vectors([" ".join(t) for t, k in zip(t2, keep) if k]); vals.append(hizli_olcu(Zv, lb))
        R["dengeli"][ad] = {k: dict(ort=float(np.mean([v[k] for v in vals])), alt=float(np.percentile([v[k] for v in vals], 2.5)), ust=float(np.percentile([v[k] for v in vals], 97.5))) for k in vals[0]}
        print("DENGELI", ad, json.dumps(R["dengeli"][ad]), flush=True)
    # en yakın komşu rastgele beklenti
    p = np.array([np.mean(lab == a) for a in ALANLAR]); R["nn_rastgele_beklenti"] = float(1 - (p ** 2).sum())
    jdump(R, "duyarlik.json")

if PARCA == "gomme":
    import re, nltk, torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    MODEL = "sentence-transformers/all-mpnet-base-v2"
    st = SentenceTransformer(MODEL, device="cpu")
    lem, wn_pos = g["lem"], g["wn_pos"]
    raw = list(df["text"].astype(str))
    # her ham sözcüğün lemması (ön işlemeyle aynı kurallar, durak sözcükler korunur)
    tagged = []
    for s in raw:
        toks = re.findall(r"[A-Za-z]+|[^A-Za-z\s]+", s)
        low = [t.lower() for t in toks]
        alpha = [t for t in low if t.isalpha()]
        tags = dict()
        pos = nltk.pos_tag(alpha); it = iter(pos)
        out = []
        for t, tl in zip(toks, low):
            if tl.isalpha():
                w, tg = next(it); out.append((t, lem.lemmatize(w, wn_pos(tg))))
            else:
                out.append((t, None))
        tagged.append(out)
    def mask_text(mask):
        res = []
        for out in tagged:
            parts = [t for t, l in out if l is None or (l not in mask and t.lower() not in mask)]
            s = " ".join(parts); s = re.sub(r"\s+([,.;:?!)])", r"\1", s); res.append(s)
        return res
    def emb_olcu(mask, perm=1000, kume=True, seed=SEED):
        E = st.encode(mask_text(mask), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        o = hizli_olcu(E, lab, perm)
        if kume: o["hdb"] = umap_hdb(E, lab, seed=seed)[0]
        cen = {a: E[lab == a].mean(0) for a in ALANLAR}
        o["merkez"] = {f"{a}-{b}": float(cosine_similarity([cen[a]], [cen[b]])[0, 0]) for a, b in itertools.combinations(ALANLAR, 2)}
        return o, E
    R = dict(model=MODEL, duzeyler={}, rastgele={})
    t0 = time.time()
    for ad, m in [("M0", set()), ("M1", DUZEY["M1"]), ("M2", DUZEY["M2"]), ("M3", DUZEY["M3"]), ("M4", DUZEY["M4"]),
                  ("C10", IADELI["C10"]), ("C25", IADELI["C25"]), ("C50", IADELI["C50"])]:
        R["duzeyler"][ad], E = emb_olcu(m)
        if ad in ("M0", "M3"):
            R["duzeyler"][ad]["_U2"] = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(E).tolist()
        print("GOMME", ad, {k: v for k, v in R["duzeyler"][ad].items() if k != "_U2"}, round(time.time() - t0), flush=True)
    R["ornek_maskeli_metin"] = {ad: mask_text(DUZEY[ad] if ad in DUZEY else set())[:3] for ad in ["M0", "M1", "M3"]}
    NR = min(TEKRAR, 30)
    for ad in ["M1", "M2", "M3", "M4"]:
        r = np.random.default_rng(SEED + 4000); rows = []
        for i in range(NR):
            o, _ = emb_olcu(rastgele_esli(DUZEY[ad], r), perm=0, kume=(i < 15), seed=SEED + i); rows.append(o)
        icd = np.array([x["ic_dis"] for x in rows]); obs = R["duzeyler"][ad]
        hd = np.array([x["hdb"]["dkb_tek_grup"] for x in rows if "hdb" in x])
        R["rastgele"][ad] = dict(n=NR, ic_dis_ort=float(icd.mean()), ic_dis_alt=float(np.percentile(icd, 2.5)), ic_dis_ust=float(np.percentile(icd, 97.5)),
                                 p_ic_dis=float((np.sum(icd <= obs["ic_dis"]) + 1) / (NR + 1)),
                                 siluet_ort=float(np.mean([x["siluet"] for x in rows])), nn_ort=float(np.mean([x["nn_disi"] for x in rows])),
                                 hdb_ort=float(hd.mean()), hdb_alt=float(np.percentile(hd, 2.5)), hdb_ust=float(np.percentile(hd, 97.5)))
        print("GOMME RASTGELE", ad, json.dumps(R["rastgele"][ad]), round(time.time() - t0), flush=True)
    jdump(R, "gomme.json")

if PARCA == "gecirgen":
    _full = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analiz2.py"), encoding="utf-8").read()
    _blk = "BANTLAR = [" + _full.split("BANTLAR = [")[1].split('R["gecirgenlik"] = {}')[0]
    exec(compile(_blk, "analiz2.py", "exec"), g)
    BANTLAR = g["BANTLAR"]; eslemeli_imza = g["eslemeli_imza"]
    idx = np.arange(len(df))
    def imza(tsel, lsel, K, oran=False):
        dfreq = collections.Counter(t for ts in tsel for t in set(ts))
        z_all = {a: log_odds(tsel, lsel, a) for a in ALANLAR}
        if not oran:
            return eslemeli_imza(z_all, dfreq, K), dfreq
        # alan içi belge oranıyla eşleme: terimin kendi alanındaki belge oranı bantlara ayrılır
        n_a = {a: int((lsel == a).sum()) for a in ALANLAR}
        dfa = {a: collections.Counter(t for ts, l in zip(tsel, lsel) if l == a for t in set(ts)) for a in ALANLAR}
        OB = [(0.03, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, 1.01)]
        cand = {a: [[w for w, v in sorted(z_all[a].items(), key=lambda x: -x[1]) if v >= 1.0 and dfreq[w] >= 3 and lo <= dfa[a][w] / n_a[a] < hi] for lo, hi in OB] for a in ALANLAR}
        m = [min(len(cand[a][b]) for a in ALANLAR) for b in range(len(OB))]
        if sum(m) <= K: n = m[:]
        else:
            raw_ = [K * x / sum(m) for x in m]; n = [int(math.floor(x)) for x in raw_]
            for b in sorted(range(len(m)), key=lambda b: -(raw_[b] - n[b])):
                if sum(n) >= K: break
                if n[b] < m[b]: n[b] += 1
        return {a: [w for b in range(len(OB)) for w in cand[a][b][:n[b]]] for a in ALANLAR}, dfreq
    def oran_hesap(sig, meas_idx):
        rate, cnt, ntoks, ndoc = {}, {}, {}, {}
        for a in ALANLAR:
            sa = set(sig[a])
            for b in ALANLAR:
                docs = [TOK[i] for i in meas_idx if lab[i] == b]; nt = sum(map(len, docs))
                c = sum(1 for ts in docs for t in ts if t in sa)
                rate[(a, b)] = c / nt * 1000 if nt else 0.0; cnt[(a, b)] = c; ntoks[b] = nt; ndoc[(a, b)] = sum(1 for ts in docs if sa & set(ts))
        return rate, cnt, ntoks, ndoc
    def ozet(res):
        summ = {k: dict(medyan=float(np.nanmedian(v)), alt=float(np.nanpercentile(v, 2.5)), ust=float(np.nanpercentile(v, 97.5))) for k, v in res.items()}
        asim = {}
        for a, b in itertools.combinations(ALANLAR, 2):
            d1 = np.array(res[f"{a}->{b}"]) - np.array(res[f"{b}->{a}"])
            asim[f"{a}->{b} eksi {b}->{a}"] = dict(medyan=float(np.nanmedian(d1)), alt=float(np.nanpercentile(d1, 2.5)), ust=float(np.nanpercentile(d1, 97.5)), pay_pozitif=float(np.nanmean(d1 > 0)))
        return summ, asim
    R = {}
    # 1. yarı bölme, K = 20, sayımlar
    for adı, oran in [("yari_E20", False), ("yari_oran_E20", True)]:
        res = {f"{a}->{b}": [] for a in ALANLAR for b in ALANLAR if a != b}; sayim = {f"{a}->{b}": [] for a in ALANLAR for b in ALANLAR}; belge = {f"{a}->{b}": [] for a in ALANLAR for b in ALANLAR}
        nsig = {a: [] for a in ALANLAR}; olcum_belge = {a: [] for a in ALANLAR}; olcum_token = {a: [] for a in ALANLAR}
        for s in range(200):
            r2 = np.random.default_rng(SEED + s); sel = np.zeros(len(df), bool)
            for a in ALANLAR:
                ia = idx[lab == a]; sel[r2.choice(ia, len(ia) // 2, replace=False)] = True
            sig, _ = imza([TOK[i] for i in idx[sel]], lab[sel], 20, oran)
            rate, cnt, nt, nd = oran_hesap(sig, idx[~sel])
            for a in ALANLAR:
                nsig[a].append(len(sig[a])); olcum_belge[a].append(int(((~sel) & (lab == a)).sum())); olcum_token[a].append(nt[a])
                for b in ALANLAR:
                    sayim[f"{a}->{b}"].append(cnt[(a, b)]); belge[f"{a}->{b}"].append(nd[(a, b)])
                    if a != b: res[f"{a}->{b}"].append(rate[(a, b)] / rate[(a, a)] if rate[(a, a)] > 0 else np.nan)
        summ, asim = ozet(res)
        R[adı] = dict(gecirgenlik=summ, asimetri=asim, sayim_ort={k: float(np.mean(v)) for k, v in sayim.items()},
                      sayim_sifir_pay={k: float(np.mean(np.array(v) == 0)) for k, v in sayim.items()},
                      belge_ort={k: float(np.mean(v)) for k, v in belge.items()}, imza_sayisi={a: float(np.mean(v)) for a, v in nsig.items()},
                      olcum_belge={a: float(np.mean(v)) for a, v in olcum_belge.items()}, olcum_token={a: float(np.mean(v)) for a, v in olcum_token.items()})
        print(adı, json.dumps(R[adı]["gecirgenlik"]), json.dumps(R[adı]["asimetri"]), flush=True)
    # 2. tekrarlı 5 katlı: imza dörtte birlik dışı katta seçilir, katın dışındaki sayımlar tüm derlemde toplanır
    res = {f"{a}->{b}": [] for a in ALANLAR for b in ALANLAR if a != b}
    for rep in range(40):
        r2 = np.random.default_rng(SEED + 500 + rep); fold = np.zeros(len(df), int)
        for a in ALANLAR:
            ia = r2.permutation(idx[lab == a]); fold[ia] = np.arange(len(ia)) % 5
        cnt_t = collections.Counter(); ntok_t = collections.Counter()
        for f in range(5):
            tr = fold != f
            sig, _ = imza([TOK[i] for i in idx[tr]], lab[tr], 20)
            rate, cnt, nt, _ = oran_hesap(sig, idx[~tr])
            for a in ALANLAR:
                for b in ALANLAR: cnt_t[(a, b)] += cnt[(a, b)]
            for b in ALANLAR: ntok_t[b] += nt[b]
        for a in ALANLAR:
            raa = cnt_t[(a, a)] / ntok_t[a]
            for b in ALANLAR:
                if a != b: res[f"{a}->{b}"].append((cnt_t[(a, b)] / ntok_t[b]) / raa)
    summ, asim = ozet(res)
    R["kat5_E20"] = dict(gecirgenlik=summ, asimetri=asim, tekrar=40)
    print("kat5", json.dumps(summ), json.dumps(asim), flush=True)
    jdump(R, "gecirgenlik_ek.json")

if PARCA == "ag":
    import networkx as nx
    toksN = [[t for t in ts if t not in ACIK_CAPA] for ts in TOK]
    docsets = [set(t) for t in toksN]; N = len(docsets)
    alan_n = {a: int((lab == a).sum()) for a in ALANLAR}
    cnt = collections.Counter(t for s in docsets for t in s)
    def kur(df_min=8, co_min=3):
        nodes = sorted(t for t, c in cnt.items() if c >= df_min); ns = set(nodes)
        co = collections.Counter()
        for s in docsets:
            for a, b in itertools.combinations(sorted(x for x in s if x in ns), 2): co[(a, b)] += 1
        G = nx.Graph(); G.add_nodes_from(nodes)
        for (a, b), c in co.items():
            if c < co_min: continue
            pmi = math.log((c * N) / (cnt[a] * cnt[b]))
            if pmi > 0:
                npmi = pmi / (-math.log(c / N))
                G.add_edge(a, b, weight=c, pmi=pmi, npmi=npmi, dist=1 / pmi)
        G.remove_nodes_from(list(nx.isolates(G)))
        return G
    def profil(v):
        p = np.array([sum(1 for s, l in zip(docsets, lab) if v in s and l == a) / alan_n[a] for a in ALANLAR])
        q = p / p.sum(); h = float(-(q[q > 0] * np.log(q[q > 0])).sum() / math.log(3))
        return p, h
    def tablo(G, rank=False):
        btw = nx.betweenness_centrality(G, weight="dist")
        comms = nx.community.louvain_communities(G, weight="weight", seed=SEED); comm_of = {v: i for i, c in enumerate(comms) for v in c}
        rows = []
        for v in G.nodes:
            p, h = profil(v)
            k = G.degree(v, weight="weight"); per = collections.Counter()
            for u in G[v]: per[comm_of[u]] += G[v][u]["weight"]
            rows.append(dict(terim=v, df=cnt[v], arasindalik=btw[v], katilim=1 - sum((x / k) ** 2 for x in per.values()), topluluk=comm_of[v],
                             VA=p[0], DP=p[1], PY=p[2], H=h, baskin=ALANLAR[int(np.argmax(p))]))
        c = pd.DataFrame(rows)
        b1, b0 = np.polyfit(np.log(c.df), c.arasindalik, 1); c["artik"] = c.arasindalik - (b0 + b1 * np.log(c.df))
        # sıra temelli artık: arasındalık sırasının log(df) sırası üzerine doğrusal artığı
        rb, rd = c.arasindalik.rank(), np.log(c.df).rank(); bb1, bb0 = np.polyfit(rd, rb, 1); c["artik_sira"] = (rb - (bb0 + bb1 * rd)) / len(c)
        return c, comms
    R = {}
    G = kur(); cen, comms = tablo(G)
    R["ag"] = dict(dugum=G.number_of_nodes(), kenar=G.number_of_edges(), topluluk=len(comms), modularite=float(nx.community.modularity(G, comms, weight="weight")))
    R["topluluklar"] = []
    for i, c in enumerate(comms):
        sub = cen[cen.topluluk == i]
        R["topluluklar"].append(dict(n=len(c), baskin_dagilim=sub.baskin.value_counts().to_dict(), alan_profil=sub[["VA", "DP", "PY"]].mean().round(4).to_dict(),
                                     terimler=sub.sort_values("df", ascending=False).terim.head(10).tolist()))
    # özvektör merkeziliği
    eig = nx.eigenvector_centrality(G, weight="weight", max_iter=5000); cen["ozvektor"] = cen.terim.map(eig)
    R["ozvektor_ilk10"] = cen.sort_values("ozvektor", ascending=False).head(10)[["terim", "ozvektor", "H"]].round(3).values.tolist()
    # alanlar arası arasındalık: yalnızca baskın alanı farklı düğüm çiftleri arasındaki en kısa yollar
    dom = dict(zip(cen.terim, cen.baskin))
    paths_btw = collections.Counter(); nodes = list(G.nodes)
    for s in nodes:
        pred, sigma, D = {}, {}, {}
        # Brandes tek kaynak (ağırlıklı)
        import heapq
        S = []; P = {v: [] for v in nodes}; sig = dict.fromkeys(nodes, 0.0); sig[s] = 1.0; Dd = {}; seen = {s: 0}; Q = [(0, s, s)]
        c_ = itertools.count(); Q = [(0, next(c_), s, s)]
        while Q:
            dist, _, pr, v = heapq.heappop(Q)
            if v in Dd: continue
            sig[v] += sig[pr] if pr != v else 0
            S.append(v); Dd[v] = dist
            for w, ed in G[v].items():
                vw = dist + ed["dist"]
                if w not in Dd and (w not in seen or vw < seen[w]):
                    seen[w] = vw; heapq.heappush(Q, (vw, next(c_), v, w)); sig[w] = 0.0; P[w] = [v]
                elif vw == seen.get(w):
                    sig[w] += sig[v]; P[w].append(v)
        delta = dict.fromkeys(S, 0.0)
        while S:
            w = S.pop()
            coeff = ((1.0 if dom[w] != dom[s] else 0.0) + delta[w]) / sig[w] if sig[w] else 0
            for v in P[w]: delta[v] += sig[v] * coeff
            if w != s: paths_btw[w] += delta[w]
    n = len(nodes); norm = 1 / ((n - 1) * (n - 2))
    cen["arasi_alan"] = cen.terim.map(lambda v: paths_btw[v] * norm)
    b1, b0 = np.polyfit(np.log(cen.df), cen.arasi_alan, 1); cen["arasi_alan_artik"] = cen.arasi_alan - (b0 + b1 * np.log(cen.df))
    R["arasi_alan_ilk12"] = cen.sort_values("arasi_alan", ascending=False).head(12)[["terim", "df", "arasi_alan", "arasi_alan_artik", "katilim", "H"]].round(4).values.tolist()
    R["spearman"] = {"arasi_alan_H": float(cen[["arasi_alan_artik", "H"]].corr("spearman").iloc[0, 1]),
                     "artik_H": float(cen[["artik", "H"]].corr("spearman").iloc[0, 1]), "artik_sira_H": float(cen[["artik_sira", "H"]].corr("spearman").iloc[0, 1])}
    secili = ["metric", "measure", "evaluate", "task", "process", "prioritize", "technique", "method", "tool", "performance"]
    R["secili_terimler"] = cen[cen.terim.isin(secili)][["terim", "df", "arasindalik", "artik", "artik_sira", "arasi_alan", "arasi_alan_artik", "katilim", "H", "topluluk"]].round(4).values.tolist()
    # eşik duyarlılığı: köprü ölçütü
    R["kopru_esik"] = []
    for hq in [0.80, 0.85, 0.90]:
        for q in [0.70, 0.75, 0.80]:
            for col in ["artik", "artik_sira", "arasi_alan_artik"]:
                thr = cen[col].quantile(q); br = cen[(cen[col] >= thr) & (cen.H >= hq)].sort_values(col, ascending=False).terim.tolist()
                R["kopru_esik"].append(dict(H=hq, q=q, olcu=col, kopru=br))
    # ağ kurma eşikleri
    R["ag_esik"] = []
    for dmin in [6, 8, 10]:
        for cmin in [2, 3, 4]:
            G2 = kur(dmin, cmin); c2, cm2 = tablo(G2)
            thr = c2.artik.quantile(0.75); br = c2[(c2.artik >= thr) & (c2.H >= 0.90)].terim.tolist()
            thr85 = c2.artik.quantile(0.75); br85 = c2[(c2.artik >= thr85) & (c2.H >= 0.85)].terim.tolist()
            karma = sum(1 for i in range(len(cm2)) if (c2[c2.topluluk == i].baskin.value_counts(normalize=True).max() < 0.6))
            R["ag_esik"].append(dict(df_min=dmin, co_min=cmin, dugum=G2.number_of_nodes(), kenar=G2.number_of_edges(), topluluk=len(cm2),
                                     modularite=float(nx.community.modularity(G2, cm2, weight="weight")), karma_topluluk=karma, kopru_H90=br, kopru_H85=br85,
                                     rho_artik_H=float(c2[["artik", "H"]].corr("spearman").iloc[0, 1])))
            print("AGESIK", dmin, cmin, R["ag_esik"][-1], flush=True)
    # NPMI ağırlıklı Louvain ile bölünme uyumu
    from sklearn.metrics import adjusted_rand_score
    cm_n = nx.community.louvain_communities(G, weight="npmi", seed=SEED); co_n = {v: i for i, c in enumerate(cm_n) for v in c}
    R["npmi_louvain"] = dict(topluluk=len(cm_n), ari=float(adjusted_rand_score([cen.set_index("terim").topluluk[v] for v in G.nodes], [co_n[v] for v in G.nodes])))
    # Louvain tohum kararlılığı
    aris = []
    for s in range(1, 21):
        cms = nx.community.louvain_communities(G, weight="weight", seed=s); cos = {v: i for i, c in enumerate(cms) for v in c}
        aris.append(adjusted_rand_score([cen.set_index("terim").topluluk[v] for v in G.nodes], [cos[v] for v in G.nodes]))
    R["louvain_tohum_ari"] = dict(ort=float(np.mean(aris)), min=float(min(aris)))
    cen.to_csv(os.path.join(OUT, "ag_merkezilik_ek.csv"), index=False)
    jdump(R, "ag_ek.json")
    # ---- Şekil 4: okunaklı ağ (en yüksek dereceli 45 düğüm, topluluk renkleri, etiket itme)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from adjustText import adjust_text
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    deg = dict(G.degree()); top = sorted(G.nodes, key=lambda v: -deg[v])[:45]
    sg = G.subgraph(top).copy(); pos = nx.kamada_kawai_layout(sg, weight=None)
    ci = cen.set_index("terim")
    pal = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000", "#2e75b6"]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    nx.draw_networkx_edges(sg, pos, ax=ax, alpha=.12, width=[0.3 + 0.06 * d["weight"] for _, _, d in sg.edges(data=True)])
    bt = cen.set_index("terim").arasindalik
    sizes = [60 + 2500 * bt[v] for v in sg.nodes]
    cols = [pal[int(ci.topluluk[v]) % len(pal)] for v in sg.nodes]
    edge = ["black" if ci.H[v] >= 0.90 else "none" for v in sg.nodes]
    nx.draw_networkx_nodes(sg, pos, ax=ax, node_size=sizes, node_color=cols, alpha=.8, edgecolors=edge, linewidths=1.2)
    texts = [ax.text(pos[v][0], pos[v][1], v, fontsize=7.5, ha="center", va="center") for v in sg.nodes]
    adjust_text(texts, ax=ax, expand=(1.25, 1.4), arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4))
    from matplotlib.lines import Line2D
    prof = {i: R["topluluklar"][i]["alan_profil"] for i in range(len(comms))}
    def tadi(i):
        b = R["topluluklar"][i]["baskin_dagilim"]; top_ = max(b, key=b.get); share = b[top_] / sum(b.values())
        nm = {"VA": "veri analitiği", "DP": "dijital pazarlama", "PY": "proje yönetimi"}
        return f"T{i + 1}: " + (f"ağırlıkla {nm[top_]}" if share >= 0.6 else "karma") + f" ({R['topluluklar'][i]['n']} terim)"
    h = [Line2D([0], [0], marker="o", ls="", markersize=7, markerfacecolor=pal[i % len(pal)], markeredgewidth=0, label=tadi(i)) for i in range(len(comms))]
    h.append(Line2D([0], [0], marker="o", ls="", markersize=7, markerfacecolor="white", markeredgecolor="black", label="Dengeli dağılım (H ≥ 0,90)"))
    for s_, l_ in [(0.01, "Arasındalık 0,01"), (0.10, "Arasındalık 0,10")]:
        h.append(Line2D([0], [0], marker="o", ls="", markersize=math.sqrt(60 + 2500 * s_), markerfacecolor="#cccccc", markeredgewidth=0, label=l_))
    ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.axis("off"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "sekil4_ag.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    assert not (set(sg.nodes) & ACIK_CAPA)

if PARCA == "izgara":
    NN, NC, MCS = [5, 10, 15, 30, 50], [2, 5, 10], [5, 10, 20, 40]
    R = {}
    for lev in ["M0", "M1", "M3"]:
        t2, keep = masked(DUZEY[lev]); lb = lab[keep]
        _, Zv, _, _ = lsa_vectors([" ".join(t) for t, k in zip(t2, keep) if k])
        rows = []
        for nn, nc in itertools.product(NN, NC):
            U = umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine", random_state=SEED).fit_transform(Zv)
            for mcs in MCS:
                cl = hdbscan.HDBSCAN(min_cluster_size=mcs).fit(U).labels_
                rows.append(dict(n_neighbors=nn, n_components=nc, min_cluster_size=mcs, **hdb_uc(cl, lb)))
        o = {}
        for k in ["dkb_tek_grup", "dkb_haric", "dkb_tekil", "gurultu", "kume", "kapsama"]:
            d = np.array([r[k] for r in rows], float)
            o[k] = dict(medyan=float(np.median(d)), q1=float(np.percentile(d, 25)), q3=float(np.percentile(d, 75)), min=float(d.min()), max=float(d.max()))
        R[lev] = dict(izgara=rows, ozet=o)
        print(lev, json.dumps(o), flush=True)
    jdump(R, "ek_izgara_uc_kural.json")
if PARCA == "tablo5":
    R = {}
    for lev in ["M0", "M1", "M3"]:
        o = tam_olcu(DUZEY[lev], n_seed=10); o.pop("lda_js", None)
        t2, keep = masked(DUZEY[lev]); lb = lab[keep]
        _, Zv, _, _ = lsa_vectors([" ".join(t) for t, k in zip(t2, keep) if k])
        dk = []
        for nn, nc in itertools.product([5, 10, 15, 30, 50], [2, 5, 10]):
            U = umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine", random_state=SEED).fit_transform(Zv)
            for mcs in [5, 10, 20, 40]:
                dk.append(hdb_uc(hdbscan.HDBSCAN(min_cluster_size=mcs).fit(U).labels_, lb)["dkb_tek_grup"])
        o["izgara_medyan"] = float(np.median(dk)); o["izgara_q1"] = float(np.percentile(dk, 25)); o["izgara_q3"] = float(np.percentile(dk, 75))
        R[lev] = o
        print(lev, o["belge"], o["ic_dis"], o["nn_disi"], o["lda_dkb"], o["hdb"], o["izgara_medyan"], flush=True)
    jdump(R, "tablo5.json")
print("bitti", PARCA)
