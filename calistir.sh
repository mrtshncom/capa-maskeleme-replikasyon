#!/usr/bin/env bash
# Tüm analizlerin sırayla yeniden üretilmesi (Python 3.13, requirements.txt).
set -euo pipefail
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
python -m nltk.downloader stopwords wordnet omw-1.4 averaged_perceptron_tagger_eng
# 1. Denetim ve temizleme
python analiz/temizlik.py veri/final_qa_db.csv veri_temiz 0.80
python analiz/hizalama_denetimi.py veri/final_qa_db.csv cikti_ek/hizalama_denetimi.json
# 2. Ana analiz (M0-M4, LDA taraması, geçirgenlik, ağ)
python analiz/analiz2.py veri_temiz/veri_temiz.csv cikti_ana
# 3. Kontrol deneyleri ve ek analizler
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek kontrol --duzey M1,M2 --belirtec
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek kontrol --duzey M3,M4 --belirtec
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek kontrol --duzey C10,C25 --tum
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek js --duzey M0,M1,M2
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek js --duzey M3,M4,C10
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek js --duzey C25,C50
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek duyarlik
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek gomme
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek gecirgen
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek ag
python analiz/analiz3.py veri_temiz/veri_temiz.csv cikti_ek izgara
python analiz/npmi_denetim.py veri_temiz/veri_temiz.csv cikti_ek
python analiz/ek_veri.py veri_temiz/veri_temiz.csv cikti_ek
# 4. Veri sürümlerine duyarlılık (ham ve yakın kopyaları korunmuş veri)
for v in ham:veri/final_qa_db.csv kopyali:veri_temiz/veri_temiz_kopyali.csv; do
  ad=${v%%:*}; yol=${v#*:}
  python analiz/analiz3.py "$yol" cikti_ek/$ad tablo5
  python analiz/analiz3.py "$yol" cikti_ek/$ad gecirgen
done
# 5. ESCO dış doğrulama (esco/esco_ham.json 20.11.2026 tarihli çekimdir, yeniden çekmek için: python analiz/esco_cek.py esco)
python analiz/esco_analiz.py esco/esco_ham.json veri_temiz/veri_temiz.csv cikti_esco
# 6. Şekiller
python analiz/sekiller.py veri_temiz/veri_temiz.csv . sekil
