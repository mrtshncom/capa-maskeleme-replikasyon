"""ESCO API'den üç alana ait mesleklerin temel ve isteğe bağlı becerilerini çeker.
Kullanım: python esco_cek.py <cikti_klasoru>"""
import json, sys, os, time, urllib.parse, urllib.request
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
API = "https://ec.europa.eu/esco/api"
MESLEK = {
 "VA": ["http://data.europa.eu/esco/occupation/d3edb8f8-3a06-47a0-8fb9-9b212c006aa2",  # data analyst
        "http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30",  # data scientist
        "http://data.europa.eu/esco/occupation/ac8b3cd1-a127-4e6a-8208-5cfcf7111955",  # statistician
        "http://data.europa.eu/esco/occupation/207d7b18-6540-432e-8aa6-785ed434572f"], # business intelligence manager
 "DP": ["http://data.europa.eu/esco/occupation/356227b9-2263-4dbd-81e9-7dfc6d3f3af7",  # online marketer
        "http://data.europa.eu/esco/occupation/6fcf4638-e7c7-4978-9302-9a7b63a3d57c",  # marketing manager
        "http://data.europa.eu/esco/occupation/092fdeeb-a76f-4ed2-b248-90fd6eaaffef",  # SEO expert
        "http://data.europa.eu/esco/occupation/137b1d11-c4bc-4016-89ba-45345d4c71f0"], # online community manager
 "PY": ["http://data.europa.eu/esco/occupation/bea99fea-0383-4c63-b944-70d4799de2c5",  # project manager
        "http://data.europa.eu/esco/occupation/8b6388a4-4904-471b-9331-d3b1211f5525",  # ICT project manager
        "http://data.europa.eu/esco/occupation/c94f3565-d987-431f-a46b-6f4e018e0bd7"], # programme manager
}
def get(path, **q):
    url = f"{API}/{path}?" + urllib.parse.urlencode(q)
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return json.load(r)
        except Exception as e:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)
meslek, beceri = {}, {}
for alan, uris in MESLEK.items():
    for u in uris:
        d = get("resource/occupation", uri=u, language="en")
        L = d["_links"]
        meslek[u] = dict(alan=alan, ad=d["title"], aciklama=d.get("description", {}).get("en", {}).get("literal", ""),
                         temel=[s["uri"] for s in L.get("hasEssentialSkill", [])], istege=[s["uri"] for s in L.get("hasOptionalSkill", [])])
        for s in L.get("hasEssentialSkill", []) + L.get("hasOptionalSkill", []):
            beceri.setdefault(s["uri"], dict(ad=s["title"]))
        print(alan, d["title"], len(meslek[u]["temel"]), len(meslek[u]["istege"]), flush=True)
for i, u in enumerate(beceri):
    d = get("resource/skill", uri=u, language="en")
    beceri[u]["aciklama"] = d.get("description", {}).get("en", {}).get("literal", "")
    beceri[u]["tur"] = d.get("_links", {}).get("hasSkillType", [{}])[0].get("title", "") if d.get("_links", {}).get("hasSkillType") else ""
    if i % 50 == 0: print(i, len(beceri), flush=True)
info = get("search", text="data analyst", type="occupation", language="en", limit=1)
json.dump(dict(meslek=meslek, beceri=beceri, cekilme_tarihi=time.strftime("%Y-%m-%d"), api=API,
               surum_bilgisi=info.get("_links", {}).get("self", {}).get("href", "")), open(os.path.join(OUT, "esco_ham.json"), "w"), ensure_ascii=False, indent=1)
print("bitti", len(beceri))
