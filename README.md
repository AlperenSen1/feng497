# feng: CWRU rulman arıza analizi

Bir CWRU rulman titreşim kaydını (`.mat`) alır, 2048 örneklik pencerelere böler, hazır
[FELA-PdM](https://huggingface.co/lowdown-labs/fela-pdm) modeline verir ve şunları yazar:

```
Dosya                   : IR007_0.mat
Sağlıksız pencere oranı : %100.0
Seviye                  : Çok güçlü arıza belirtisi
En olası arıza türü     : İç halka [sağlıksız pencerelerdeki payı: %100.0]
```

Bu bir simülasyon/demo projesidir. Gerçek sensör verisiyle test edilmedi.

## Nasıl çalışır

1. `.mat` dosyasından sürücü ucu (`DE_time`) titreşim sinyali okunur.
2. Sinyal 2048 örneklik, 1024 kaydırmalı pencerelere bölünür.
3. Model her pencereye 10 sınıftan birini verir (sağlam + 3 arıza türü x 3 hasar boyutu).
4. Sınıf 0 (sağlam) olmayan pencerelerin oranı hesaplanır.
5. Orana göre seviye belirlenir. En olası arıza türü (iç halka, bilya, dış halka), oran
   %30 ve üzerindeyse yazılır. Hasar boyutları türde birleştirilir.

| Sağlıksız pencere oranı | Seviye |
|---|---|
| %0–10 | Normal |
| %10–30 | İzlenmeli (hafif belirti) |
| %30–60 | Olası arıza |
| %60–90 | Güçlü arıza belirtisi |
| %90 ve üstü | Çok güçlü arıza belirtisi |

Bu bantlar örnek değerlerdir, veriyle doğrulanmadı. Oran bir arıza olasılığı ya da kalan ömür
tahmini değildir, modelin pencerelerin yüzde kaçında anormallik gördüğünü gösterir.

## Kurulum (Windows, CMD)

Python 3.12 gerekir.

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install numpy scipy safetensors huggingface_hub
```

Modeli indir (`fela-pdm` klasörüne):

```bat
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('lowdown-labs/fela-pdm', local_dir='fela-pdm')"
```

CWRU verisini resmi siteden indir: <https://engineering.case.edu/bearingdatacenter>
(12 kHz sürücü ucu ve normal veri sayfaları). Dosyalar `https://engineering.case.edu/sites/default/files/<numara>.mat`
adresinden indirilir. Program `DE_time` sinyali içeren herhangi bir CWRU `.mat` dosyasını okur.
Eğitim setiyle aynı 40 dosyayı kullanmak için (dış bilezik: saat 6 konumu):

| Sınıf | Yük 0, 1, 2, 3 dosya numaraları |
|---|---|
| Normal | 97–100 |
| İç bilezik 0.007 / 0.014 / 0.021 | 105–108 / 169–172 / 209–212 |
| Bilya 0.007 / 0.014 / 0.021 | 118–121 / 185–188 / 222–225 |
| Dış bilezik 0.007 / 0.014 / 0.021 | 130–133 / 197–200 / 234–237 |

## Kullanım

```bat
feng run
```

Program "Dosya yolu:" diye sorar, yolu ayrı satıra yazarsın. Doğrudan da verilebilir:

```bat
feng run C:\yol\dosya.mat
```

(`feng.bat` proje klasöründeyken çalışır. Aynı iş için:
`.venv\Scripts\python.exe predict_file.py [dosya.mat]`.)

## Sınırlılıklar

- Model CWRU ile eğitildi. CWRU'daki yaygın pencere bölme yönteminde aynı kayıttan pencereler hem
  eğitimde hem testte yer alır, bu yüzden CWRU üzerindeki neredeyse kusursuz sonuçlar iyimserdir ve
  başarı kanıtı sayılmamalıdır. Farklı yük veya hasar boyutuna genelleme test edilmedi.
- Modelin "bilinmeyen" sınıfı yoktur. Saf rastgele gürültüyü bile yüksek güvenle bir arıza
  sınıfına atadığını gözlemledik. Bu yüzden modelin güven değeri tek başına güvenilir değildir.
- Normal kayıtlar arıza kayıtlarından 2–4 kat uzundur ve 48 kHz'de kaydedilmiş olabilir
  (doğrulanmadı). Doğruysa model "sağlam"ı kayıt farkından da ayırıyor olabilir.
- Gerçek sensör verisi kullanılmadı. Bu bir güvenlik denetleyicisi ya da bakım kararı için tek
  başına dayanak değildir.

## Lisans ve atıflar

- Model: FELA-PdM, Lowdown Labs, "Lowdown Labs Lovely License 1.0" (CC BY-NC 4.0 + Hippocratic
  License 3.0). Ticari kullanım yoktur. Model bu depoda dağıtılmaz.
- Veri: Case Western Reserve University Bearing Data Center. Veri setinde açık bir kullanım
  lisansı yayımlanmamıştır, bu depoda dağıtılmaz.
