# Soru-cevap derleminde alan ayrışmasının çapa terim maskelemesiyle sınanması: yeniden üretim paketi

Bu depo, makaledeki tüm sayıların, tabloların ve şekillerin yeniden üretilmesi için gereken veriyi, kodu ve çıktıları içermektedir.

## Veri
- `veri/final_qa_db.csv`: Hugging Face `aumghag/Data-Analytics-Digital-Marketing-Project-Management-QA_DB` veri seti, kayıt `6d0d14fa8c37c44b425de99e303d79dcfb7ab727`, Apache-2.0 lisansı. SHA-256: `342c016f89b24670d53f800acc8e36258c3d70c7e9d4f505cd4a79c22b3fd6ab`.
- `veri_temiz/temizlik_kaydi.csv`: 489 satırın her biri için karar (korundu, onarıldı, çıkarıldı), kullanılan cevap satırı, benzerlik değerleri ve kopya bilgisi.
- `veri_temiz/veri_temiz.csv`: Analiz derlemi (434 belge). `veri_temiz_kopyali.csv`: yakın kopyaları korunmuş sürüm (454 belge).
- `veri_temiz/yakin_kopya_ciftleri.csv`: Kosinüs benzerliği 0,80 ve üzerindeki 22 çift (Ek Tablo 3).
- `esco/esco_ham.json`: ESCO programlama arayüzünden 20.11.2026 tarihinde çekilen 11 meslek ve 374 beceri.

## Kod (`analiz/`)
| Dosya | İşlev |
|---|---|
| `alan_atama.py` | Satır aralıklarına göre alan etiketleri (VA, DP, PY) |
| `temizlik.py` | Hizalama taraması, satır düzeyindeki onarım kararları, birebir ve yakın kopya ayıklama |
| `hizalama_denetimi.py` | Elle verilen hizalama kararlarının algoritmik ikinci kodlayıcıyla denetimi |
| `analiz2.py` | Ön işleme, log-odds, M0–M4 maskeleme, LDA taraması, UMAP ve HDBSCAN, frekansı eşlenmiş geçirgenlik, eş-geçim ağı |
| `analiz3.py` | İadeli (C10–C50) ve kümülatif maskeleme, eşlenmiş rastgele maskeleme, rastgele belirteç silme, Jensen-Shannon farkı, sabit örneklem, GAA boyutu, cümle gömmeleri, beş katlı geçirgenlik, ağ eşikleri, 60 ayarlık ızgara |
| `ek_izgara.py` | Ana analizdeki ızgara (tek grup kuralı) |
| `npmi_denetim.py` | Normalleştirilmiş NKB ağırlıklı Louvain denetimi |
| `ek_veri.py` | Belge başına sözlük biçimi istatistikleri, durak sözcük ve maskeleme listeleri |
| `esco_cek.py`, `esco_analiz.py` | ESCO dış doğrulama derleminin çekilmesi ve analizi |
| `sekiller.py` | Şekil 1–4 |

Çalıştırma sırası `calistir.sh` dosyasındadır. Sürümler `requirements.txt` dosyasında sabitlenmiştir (Python 3.13). Tohum taramaları dışındaki tüm rastgele süreçlerde tohum değeri 42'dir. Cümle gömmeleri `sentence-transformers/all-mpnet-base-v2` modeliyle hesaplanmıştır.

## Çıktılar
- `cikti_ana/sonuclar.json`: Ana analizin tüm sayısal değerleri (Tablo 2–4, 8–10).
- `cikti_ek/`: Kontrol deneyleri (`kontrol_*.json`), Jensen-Shannon farkı (`js_*.json`), duyarlılık (`duyarlik.json`), gömmeler (`gomme.json`), geçirgenlik (`gecirgenlik_ek.json`), ağ (`ag_ek.json`, `ag_merkezilik_ek.csv`), ızgara (`ek_izgara_uc_kural.json`), veri sürümleri (`ham/`, `kopyali/`).
- `cikti_esco/`: ESCO derlemi ve sonuçları.
- `sekil/`: Makaledeki şekiller (300 dpi).
- `ek_materyal/`: Ek 1 (Veri Seti Kartı, makalenin sonundaki ekin ayrı kopyası) ve Ek Materyal (Ek Tablolar, Ek Tablo 1'deki 60 ayarlık ızgara dahil).

## Lisans notu
Ham veri özgün Apache-2.0 lisansıyla dağıtılmaktadır. ESCO verileri Avrupa Komisyonu'nun ESCO kullanım koşullarına tabidir.

---

**English summary.** Replication package for the anchor-term masking study of a question-answer corpus (data analytics, digital marketing, project management). Run `calistir.sh` in a Python 3.13 environment built from `requirements.txt`. The raw data file is identical to the Hugging Face dataset at the commit and SHA-256 given above. `veri_temiz/temizlik_kaydi.csv` documents every row-level cleaning decision.
