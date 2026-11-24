# Rulman Arıza Kontrolü

Bu program, bir rulmanın titreşim kaydına bakar ve rulmanda arıza belirtisi olup olmadığını söyler.

Windows bilgisayarlarda çalışır. Programı size veren kişi kurulumu yaptıysa aşağıdaki adımlar yeterlidir.

> **Not:** Bu program bir deneme ve gösteri çalışmasıdır. Sonuçlar tek başına bir bakım kararı vermek için kullanılmamalıdır.

## Başlamadan önce: 2 şeyi indirin

Programın çalışması için bir yapay zekâ modeline ve kontrol edilecek kayıt dosyalarına ihtiyaç vardır. İkisini de internetten indirmeniz gerekir. Model dosyalarını bir kez indirmeniz yeterlidir.

### A. Yapay zekâ modeli

1. Şu sayfayı açın: <https://huggingface.co/lowdown-labs/fela-pdm>
2. Sayfadaki **Files and versions** (veya **Files**) sekmesine tıklayın.
3. Şu 3 dosyayı tek tek indirin (her dosyanın yanında aşağı ok işareti vardır):
   - `config.json`
   - `modeling.py`
   - `cwru.safetensors`
4. Program klasörünün içinde `fela-pdm` adında yeni bir klasör açın ve bu 3 dosyayı içine koyun.

### B. Kontrol edilecek kayıt dosyaları

1. Şu siteyi açın: <https://engineering.case.edu/bearingdatacenter>
2. Sitedeki **12k Drive End Bearing Fault Data** (arızalı rulman kayıtları) veya **Normal Baseline Data** (sağlam rulman kayıtları) sayfasını açın.
3. Tablodaki dosya adlarına (örneğin `IR007_0`) tıklayarak dosyaları indirin. İnen dosyanın adı `105.mat` gibi bir sayı olabilir, sorun değildir.

Dosyaları istediğiniz klasöre indirebilirsiniz. 3. adımda dosyayı programa kendiniz göstereceksiniz.

## Kullanmak için 4 adım

### 1. Siyah komut penceresini açın

1. Program klasörünü açın. İçinde `feng.bat` dosyasını göreceksiniz.
2. Klasörün üstündeki adres çubuğuna (klasör yolunun yazdığı yere) tıklayın.
3. `cmd` yazın ve **Enter** tuşuna basın.

Siyah bir pencere açılır. Bu pencere doğru klasörde açılmıştır.

### 2. Programı başlatın

Siyah pencereye şunu yazın ve **Enter** tuşuna basın:

```
feng run
```

Ekranda `Dosya yolu:` yazısı çıkar. İlk seferde birkaç saniye beklemeniz gerekebilir.

### 3. Kontrol edilecek dosyayı gösterin

Kontrol etmek istediğiniz kayıt dosyasını (`.mat` ile biten dosya) fare ile tutup siyah pencerenin içine bırakın. Dosyanın adresi kendiliğinden yazılır. Sonra **Enter** tuşuna basın.

### 4. Sonucu okuyun

Birkaç saniye içinde şuna benzer bir sonuç çıkar:

```
Dosya                   : IR007_0.mat
Sağlıksız pencere oranı : %100.0
Seviye                  : Çok güçlü arıza belirtisi
En olası arıza türü     : İç halka [sağlıksız pencerelerdeki payı: %100.0]
```

| Satır | Ne demek |
|---|---|
| **Dosya** | Kontrol ettiğiniz dosyanın adı. |
| **Sağlıksız pencere oranı** | Kayıt küçük parçalara bölünüp tek tek incelenir. Bu sayı, parçaların yüzde kaçında arıza belirtisi görüldüğünü gösterir. Sayı yükseldikçe belirti artar. Bu bir "arıza olma ihtimali" değildir. |
| **Seviye** | Sonucun kısa özeti. Aşağıdaki tabloya bakın. |
| **En olası arıza türü** | Belirti varsa, rulmanın hangi bölümünde olabileceğini söyler: iç halka, bilya veya dış halka. Köşeli parantezdeki yüzde, belirti gösteren parçaların ne kadarının bu türü işaret ettiğini gösterir. Yüzde yükseldikçe sonuç daha tutarlıdır. |

Bu son satır sadece seviye **"Olası arıza"** veya daha yüksekse görünür.

**Seviyeler ne anlama gelir?**

| Seviye | Anlamı |
|---|---|
| Normal | Arıza belirtisi görülmedi. |
| İzlenmeli (hafif belirti) | Hafif belirtiler var. Takip edilmesi iyi olur. |
| Olası arıza | Arıza olabilir. Kontrol edilmesi önerilir. |
| Güçlü arıza belirtisi | Belirtiler belirgin. Kontrol edilmesi önerilir. |
| Çok güçlü arıza belirtisi | Belirtiler çok belirgin. Kontrol edilmesi önerilir. |

## Başka bir dosyayı kontrol etmek

2. adımdan tekrar başlayın. Kolaylık için klavyedeki **yukarı ok** tuşuna basıp **Enter**'a basarsanız `feng run` yeniden yazılır.

## Bir sorun olursa

| Ekranda ne yazıyor | Ne yapmalı |
|---|---|
| `feng` tanınmıyor gibi bir yazı | Siyah pencere program klasöründe açılmamıştır. 1. adımı tekrar yapın. |
| Dosya bulunamadı | Dosyayı siyah pencereye yeniden sürükleyip bırakın. |
| Dosyayla ilgili başka bir hata | Bu program yalnızca rulman titreşim kaydı olan `.mat` dosyalarını okuyabilir. Başka bir dosya deneyin. |

## Kaynaklar

Bu program, Case Western Reserve University'nin rulman titreşim verisi ve Lowdown Labs'in FELA-PdM yapay zekâ modeli ile çalışır. Ticari amaçla kullanılamaz.
