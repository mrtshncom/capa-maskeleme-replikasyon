"""Soru-cevap hizalama denetimi ve veri temizleme.

Kullanım: python3 temizlik.py <final_qa_db.csv> <cikti_klasoru>

Aşama 1 (tarama): Her soru, kendi cevabıyla ve ±3 satır içindeki cevaplarla TF-IDF kosinüs
benzerliği üzerinden karşılaştırılır. Kendi benzerliği 0,10'un altında kalan ya da komşu bir
cevapla benzerliği kendi benzerliğinden en az 0,15 yüksek olan satırlar "şüpheli" işaretlenir.
Aşama 2 (elle denetim): Şüpheli satırların tamamı okunmuş ve aşağıdaki kararlar verilmiştir.
Sistematik hatalar üç ardışık bölümde yoğunlaşmaktadır; diğer şüpheli satırlar okumada doğru
hizalı bulunmuş ve korunmuştur.
Aşama 3: Birebir yinelenen soru-cevap çiftleri çıkarılır. Ardından kalan belgelerin tamamı (alan ayrımı
yapılmadan) ikişerli karşılaştırılır: her belge, lemmatize edilmiş soru + cevap metninin TF-IDF vektörüyle
(İngilizce durak sözcükler çıkarılmış, alt doğrusal terim sıklığı) temsil edilir ve kosinüs >= ESIK_KOPYA olan
çiftlerde satır sırasına göre sonra gelen belge çıkarılır (her yakın kopya kümesinden ilk satır tutulur).
"""
import re, sys, os, json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

CSV, OUT = sys.argv[1], sys.argv[2]
ESIK_KOPYA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
os.makedirs(OUT, exist_ok=True)
lem = WordNetLemmatizer()
tok = lambda s: " ".join(lem.lemmatize(w) for w in re.findall(r"[a-z]+", s.lower()))

d = pd.read_csv(CSV)
d["question"] = d["question"].astype(str).str.strip()
d["answer"] = d["answer"].astype(str).str.strip()
n = len(d)
assert (d["Unnamed: 0"].values == np.arange(n)).all()

# ---------------------------------------------------------------- Aşama 1: tarama
v = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(pd.concat([d.question, d.answer]).map(tok))
Q = v.transform(d.question.map(tok)); A = v.transform(d.answer.map(tok))
S = cosine_similarity(Q, A)
own = np.diag(S).copy()
komsu_max = np.array([max(S[i, j] for j in range(max(0, i - 3), min(n, i + 4)) if j != i) for i in range(n)])
supheli = (own < 0.10) | (komsu_max - own >= 0.15)

# ---------------------------------------------------------------- Aşama 2: elle denetim kararları
# Q satırı -> kullanılacak cevabın satırı (None: soru çıkarılır)
ESLEME = {i: i for i in range(n)}
# Bölüm 1 (kümeleme geçerlik ölçütleri): cevaplar bir satır kaymış; 114. sorunun cevabı ayırt edilemiyor.
ESLEME[114] = None
for i in range(115, 122):
    ESLEME[i] = i - 1
# Bölüm 2 (kümeleme soruları): 122-144 numaralı sorulara karar ağacı cevapları eklenmiş; özgün cevaplar yok.
for i in range(122, 145):
    ESLEME[i] = None
# Aynı bölümde 121-125 numaralı satırlardaki cevaplar 145-149 numaralı karar ağacı sorularına aittir.
for i in range(145, 150):
    ESLEME[i] = i - 24
# Bölüm 3 (dijital pazarlama): cevaplar bir satır kaymış; 288. sorunun cevabı veri setinde yok.
ESLEME[288] = None
for i in range(289, 322):
    ESLEME[i] = i - 1

rows = []
for i in range(n):
    j = ESLEME[i]
    karar = "cikarildi" if j is None else ("onarildi" if j != i else "korundu")
    rows.append(dict(satir=i, supheli=bool(supheli[i]), karar=karar, cevap_satiri=("" if j is None else j),
                     ozgun_benzerlik=round(float(own[i]), 3), yeni_benzerlik=("" if j is None else round(float(S[i, j]), 3)),
                     soru=d.question[i], ozgun_cevap=d.answer[i], kullanilan_cevap=("" if j is None else d.answer[j])))
log = pd.DataFrame(rows)

clean = log[log.karar != "cikarildi"][["satir", "soru", "kullanilan_cevap", "karar"]].rename(
    columns={"satir": "Unnamed: 0", "soru": "question", "kullanilan_cevap": "answer"}).reset_index(drop=True)

# ---------------------------------------------------------------- Aşama 3: yinelenen ve yakın kopyalar
exact = clean.duplicated(["question", "answer"])
clean_e = clean[~exact].reset_index(drop=True)
txt = (clean_e.question + " " + clean_e.answer).map(tok)
X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(txt)
C = cosine_similarity(X); np.fill_diagonal(C, 0)
keep = np.ones(len(clean_e), bool); kopya_of = {}
for i in range(len(clean_e)):
    if not keep[i]:
        continue
    for j in np.where(C[i] >= ESIK_KOPYA)[0]:
        if j > i and keep[j]:
            keep[j] = False; kopya_of[int(clean_e.loc[j, "Unnamed: 0"])] = int(clean_e.loc[i, "Unnamed: 0"])
final = clean_e[keep].reset_index(drop=True)
# Eşik üstündeki tüm belge çiftleri (alan sınırı gözetilmeden, tüm çiftler karşılaştırılmıştır)
iu = np.triu_indices(len(clean_e), 1)
ust = C[iu] >= ESIK_KOPYA
satir = clean_e["Unnamed: 0"].values
def alan_ad(r):
    return "VA" if r <= 257 else ("DP" if r <= 374 or 405 <= r <= 419 else "PY")
ciftler = pd.DataFrame(dict(satir_1=satir[iu[0][ust]], satir_2=satir[iu[1][ust]], kosinus=np.round(C[iu][ust], 3)))
ciftler["alan_1"] = ciftler.satir_1.map(alan_ad); ciftler["alan_2"] = ciftler.satir_2.map(alan_ad)
ciftler.to_csv(os.path.join(OUT, "yakin_kopya_ciftleri.csv"), index=False, encoding="utf-8-sig")
log["yakin_kopya_ilk_satir"] = log.satir.map(kopya_of).fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
log["birebir_yinelenen"] = log.satir.isin(clean[exact]["Unnamed: 0"])
log["analize_alindi"] = log.satir.isin(final["Unnamed: 0"])
log.to_csv(os.path.join(OUT, "temizlik_kaydi.csv"), index=False, encoding="utf-8-sig")
final.to_csv(os.path.join(OUT, "veri_temiz.csv"), index=False, encoding="utf-8")
clean_e.to_csv(os.path.join(OUT, "veri_temiz_kopyali.csv"), index=False, encoding="utf-8")

def seg(rr):
    return {"VA": int(((rr >= 0) & (rr <= 257)).sum()), "DP": int((((rr >= 258) & (rr <= 374)) | ((rr >= 405) & (rr <= 419))).sum()),
            "PY": int((((rr >= 375) & (rr <= 404)) | (rr >= 420)).sum())}
fin_sim = [S[i, ESLEME[i]] for i in final["Unnamed: 0"]]
ozet = dict(ham=n, supheli=int(supheli.sum()), supheli_dogru_bulunan=int((supheli & (log.karar == "korundu")).sum()),
            korundu=int((log.karar == "korundu").sum()), onarildi=int((log.karar == "onarildi").sum()),
            cikarildi=int((log.karar == "cikarildi").sum()), birebir_yinelenen=int(exact.sum()),
            yakin_kopya=int((~keep).sum()), esik_kopya=ESIK_KOPYA, analiz=int(len(final)),
            analiz_alan=seg(final["Unnamed: 0"].values), kopyali_alan=seg(clean_e["Unnamed: 0"].values),
            yakin_kopya_alan=seg(np.array(list(kopya_of.keys()))), esik_ustu_cift=int(len(ciftler)),
            esik_ustu_cift_alanlar_arasi=int((ciftler.alan_1 != ciftler.alan_2).sum()),
            karsilastirilan_belge=int(len(clean_e)), karsilastirilan_cift=int(len(iu[0])),
            benzerlik_ham_ort=float(own.mean()), benzerlik_temiz_ort=float(np.mean(fin_sim)),
            benzerlik_ham_dusuk=int((own < 0.10).sum()), benzerlik_temiz_dusuk=int((np.array(fin_sim) < 0.10).sum()))
json.dump(ozet, open(os.path.join(OUT, "temizlik_ozet.json"), "w"), indent=1)
print(json.dumps(ozet, indent=1))
