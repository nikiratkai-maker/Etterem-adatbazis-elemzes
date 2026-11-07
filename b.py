import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def atlag_szamitas(szam_lista):
    '''Kiszámolja egy számsorozat átlagát a NumPy segítségével.'''
    if not szam_lista:
        return 0.0
    tomb = np.array(szam_lista)
    return float(np.mean(tomb))


# 1. ADATOK LEKÉRÉSE A KAGGLE-RŐL
print('Adatok letöltése...')
mappa = kagglehub.dataset_download('srisyra02/zomato-market-analysis')

# Megkeressük a .csv végű fájlt (3. fejezet: endswith metódus)
csv_fajl = ''
for fajl in os.listdir(mappa):
    if fajl.endswith('.csv'):
        csv_fajl = mappa + '/' + fajl
        break

# Táblázat beolvasása Pandas-szal
df = pd.read_csv(csv_fajl, encoding='latin-1')

# Oszlopok kinyerése sima Python listákba (4. fejezet: list konverzió)
arak = list(df['Price range'])
szallitasok = list(df['Has Online delivery'])
ertekelesek = list(df['Aggregate rating'])
konyhak = list(df['Cuisines'])
foglalasok = list(df['Has Table booking'])
szavazatok = list(df['Votes'])
orszagok = list(df['Country Code'])
koltsegek = list(df['Average Cost for two'])



# 1. KÉRDÉS:
# Hogyan változik az éttermek átlagos értékelése az árkategóriák (1-4) szerint,
# és van-e különbség az online kiszállítást vállaló ('Yes') és nem vállaló ('No')
# helyek között?


print('\n 1. KÉRDÉS EREDMÉNYEI ')

kategoriak = [1, 2, 3, 4]
atlag_van_szallitas = []
atlag_nincs_szallitas = []

for kat in kategoriak:
    # Listaértelmezés szűréssel és zip-pel (5. és 6. fejezet)
    ertek_igen = [ert for ar, szall, ert in zip(arak, szallitasok, ertekelesek)
                  if ar == kat and szall == 'Yes' and ert > 0]
    
    ertek_nem = [ert for ar, szall, ert in zip(arak, szallitasok, ertekelesek)
                 if ar == kat and szall == 'No' and ert > 0]

    # Saját függvény hívása (7. fejezet)
    atlag_i = atlag_szamitas(ertek_igen)
    atlag_n = atlag_szamitas(ertek_nem)

    atlag_van_szallitas.append(atlag_i)
    atlag_nincs_szallitas.append(atlag_n)

    # Formázott kiírás f-stringgel (2. és 3. fejezet)
    print(f'{kat}. árkategória -> Kiszállítással: {atlag_i:.2f} '
          f'({len(ertek_igen)} db), Nélküle: {atlag_n:.2f} ({len(ertek_nem)} db)')



# 2. KÉRDÉS:
# Melyik az 5 leggyakoribb fő konyhatípus, mekkora az átlagos értékelésük,
# és milyen arányban kínálnak asztalfoglalást ('Yes')?


print('\n 2. KÉRDÉS EREDMÉNYEI ')

# Gyakoriságszámlálás szótárral és get-tel (5. fejezet 14. dia "iskolapélda")
gyakorisag = {}
for konyha, ert in zip(konyhak, ertekelesek):
    if ert > 0 and type(konyha) == str:
        # A vesszővel elválasztott konyhákból az elsőt vesszük (3. fejezet: split, strip)
        fo_konyha = konyha.split(',')[0].strip()
        gyakorisag[fo_konyha] = gyakorisag.get(fo_konyha, 0) + 1

# Szótár rendezése érték szerint csökkenőbe lambdával, szeleteléssel az első 5 (5. és 6. fejezet)
top5_parok = sorted(gyakorisag.items(), key=lambda kv: kv[1], reverse=True)[:5]

top5_nevek = []
top5_darabok = []

# Kicsomagolás enumerate-tel, 1-től számozva (6. fejezet 19. dia)
for sorszam, (nev, db) in enumerate(top5_parok, start=1):
    top5_nevek.append(nev)
    top5_darabok.append(db)

    konyha_ert = []
    foglalas_db = 0

    for k, ert, fogl in zip(konyhak, ertekelesek, foglalasok):
        if ert > 0 and type(k) == str:
            elso = k.split(',')[0].strip()
            if elso == nev:
                konyha_ert.append(ert)
                if fogl == 'Yes':
                    foglalas_db += 1

    k_atlag = atlag_szamitas(konyha_ert)
    foglalas_arany = foglalas_db / db

    print(f'{sorszam}. {nev:<14} | Darab: {db:4} | '
          f'Átlag: {k_atlag:.2f} | Asztalfoglalás: {foglalas_arany:.1%}')



# 3. KÉRDÉS:
# Mennyi az összes adatbázisban szereplő étterem átlagos értékelése és szórása?


print('\n 3. KÉRDÉS EREDMÉNYEI  ')
ervenyes_ert = np.array([e for e in ertekelesek if e > 0])
print(f'Összes értékelt étterem átlaga (NumPy): {np.mean(ervenyes_ert):.2f}')
print(f'Értékelések szórása (NumPy): {np.std(ervenyes_ert):.2f}')



# 4. KÉRDÉS:
# Mennyi a szavazatok mediánja az adatbázisban, és mennyi a legtöbb szavazat,
# amit egyetlen étterem kapott?


print('\n 4. KÉRDÉS EREDMÉNYEI ')
szavazat_tomb = np.array(szavazatok)
print(f'Szavazatok mediánja (NumPy): {np.median(szavazat_tomb):.1f}')
print(f'Legtöbb szavazat egy étteremre (NumPy): {np.max(szavazat_tomb)}')



# 5. KÉRDÉS:
# Mennyi a két főre jutó átlagos költség (Average Cost for two) az adatbázisban?


print('\n 5. KÉRDÉS EREDMÉNYEI ')
koltseg_tomb = np.array([k for k in koltsegek if k > 0])
print(f'Két főre jutó átlagos költség (NumPy): {np.mean(koltseg_tomb):.2f}')
print(f'Két főre jutó költségek mediánja (NumPy): {np.median(koltseg_tomb):.2f}')



# GRAFIKONOK KIRAJZOLÁSA (MATPLOTLIB)


plt.figure(figsize=(12, 5))

# 1. Grafikon: Vonaldiagram az 1. kérdéshez
plt.subplot(1, 2, 1)
plt.plot(kategoriak, atlag_van_szallitas, marker='o', label='Van online szállítás')
plt.plot(kategoriak, atlag_nincs_szallitas, marker='s', label='Nincs online szállítás')
plt.title('Átlagos értékelés árkategória szerint')
plt.xlabel('Árkategória (1 = legolcsóbb, 4 = legdrágább)')
plt.ylabel('Átlagos értékelés')
plt.xticks(kategoriak)
plt.legend()
plt.grid(True)

# 2. Grafikon: Oszlopdiagram a 2. kérdéshez
plt.subplot(1, 2, 2)
plt.bar(top5_nevek, top5_darabok)
plt.title('Top 5 leggyakoribb fő konyhatípus')
plt.xlabel('Konyha típusa')
plt.ylabel('Éttermek száma (db)')
plt.grid(True)

plt.tight_layout()
plt.show()
