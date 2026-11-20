"""Elle verilen hizalama kararlarının algoritmik ikinci kodlayıcıyla karşılaştırılması.

Kullanım: python3 hizalama_denetimi.py <final_qa_db.csv> <cikti.json>
Algoritmik kodlayıcı: her soru için tüm 489 cevap arasında TF-IDF kosinüs benzerliği en yüksek cevap satırı (pencere sınırı yok).
Onarılan 45 soruda elle seçilen cevabın bu en yüksek satırla aynı olup olmadığı ve korunan satırlarda kendi cevabın en yüksek olup olmadığı raporlanır.
"""
import sys, json, re, tempfile
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
CSV, OUT = sys.argv[1], sys.argv[2]
src = open(__file__.replace("hizalama_denetimi.py", "temizlik.py"), encoding="utf-8").read()
g = {}; exec(src.split("rows = []")[0].replace("CSV, OUT = sys.argv[1], sys.argv[2]", f"CSV, OUT = {CSV!r}, {tempfile.mkdtemp()!r}"), g)
S, E, n = g["S"], g["ESLEME"], g["n"]
best = S.argmax(1); rank = lambda i, j: int((S[i] > S[i, j]).sum()) + 1
onar = [i for i in range(n) if E[i] is not None and E[i] != i]
cik = [i for i in range(n) if E[i] is None]
kor = [i for i in range(n) if E[i] == i]
res = dict(onarilan=len(onar), onarilan_en_yuksek_ayni=int(sum(best[i] == E[i] for i in onar)),
           onarilan_ilk3=int(sum(rank(i, E[i]) <= 3 for i in onar)),
           onarilan_sira_ortanca=float(np.median([rank(i, E[i]) for i in onar])),
           korunan=len(kor), korunan_kendi_en_yuksek=int(sum(best[i] == i for i in kor)),
           korunan_kendi_ilk3=int(sum(rank(i, i) <= 3 for i in kor)),
           cikarilan=len(cik), cikarilan_en_yuksek_benzerlik_ortanca=float(np.median([S[i].max() for i in cik])),
           cikarilan_en_yuksek_benzerlik_maks=float(max(S[i].max() for i in cik)),
           cikarilan_en_yuksek_baska_soruya_atanmis=int(sum(best[i] in set(E[j] for j in range(n) if E[j] is not None) for i in cik)))
# uzak kaymaların taranması: kendi cevabı ilk 3'te olmayan ve en yüksek benzerliği ±3 penceresi dışında kalan korunan satırlar
uzak = [i for i in kor if rank(i, i) > 3 and abs(int(best[i]) - i) > 3]
res["korunan_uzak_aday"] = len(uzak); res["korunan_uzak_aday_satirlar"] = uzak
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res))
