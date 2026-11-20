# Alan ataması: korpus, alanlara göre ardışık bölümler hâlinde derlenmiştir.
# Bölüm sınırları soru dizisinin okunmasıyla belirlenmiştir (özgün satır numarası, 'Unnamed: 0'):
#   0-257   Veri analitiği (VA)
#   258-374 Dijital pazarlama (DP)
#   375-404 Proje yönetimi (PY)
#   405-419 Dijital pazarlama (DP)
#   420-488 Proje yönetimi (PY)
BOLUMLER = [(0, 257, "VA"), (258, 374, "DP"), (375, 404, "PY"), (405, 419, "DP"), (420, 488, "PY")]

def assign_domains(df):
    out = []
    for i in df["Unnamed: 0"]:
        for a, b, lab in BOLUMLER:
            if a <= i <= b:
                out.append(lab); break
        else:
            out.append("BELIRSIZ")
    return out
