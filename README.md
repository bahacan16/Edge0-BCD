<div align="center">

# Edge0 for iPhone

**edge0'ın modellerini iPhone'da, cihaz üzerinde çalıştıran SwiftUI uygulaması.**

</div>

Bu depo iki şey içerir:

1. **`ios/`** — Apple'ın MLX Swift kütüphanesi üzerinde çalışan, edge0'ın
   yayınladığı modelleri telefonda çalıştıran bir iOS uygulaması.
2. **Kök dizindeki Python kaynağı** — yukarı akış
   [`Edge0-AI/edge0`](https://github.com/Edge0-AI/edge0) projesinin kopyası
   (Apache-2.0). Bu kod **macOS + Apple Silicon** içindir, telefonda çalışmaz;
   referans olarak duruyor. Onun kendi README'si
   [`README_edge0_upstream.md`](README_edge0_upstream.md) dosyasındadır.

## Neden ayrı bir uygulama?

Yukarı akış edge0 bir **Python paketi**: MLX'in Python sürümüne (`mlx-metal`)
dayanır ve yalnızca macOS'ta çalışır. iOS'ta Python yorumlayıcısı gömmek
mümkündür (bkz. `PythonKit` + `python-apple-support`), ama `mlx` saf Python
değil — derlenmiş C++/Metal çekirdekleri olan, sadece macOS için yayınlanmış
bir uzantı. Bu yüzden uygulama, Apple'ın aynı MLX motorunun **resmi Swift
portunu** (`mlx-swift` / `mlx-swift-lm`) kullanır ve edge0'ın modellerini o
motorda çalıştırır.

## Ne çalışıyor

| | edge0-8b | edge0-35b |
|---|---|---|
| Mimari | Ling 3.0 `bailing_hybrid` | Qwen3.5 MoE `qwen3_5_moe` |
| Swift tarafı | Bu depoda elle portlandı | mlx-swift-lm'de zaten var |
| İndirme | ~4.2 GB | ~23 GB |
| Bellek stratejisi | Tamamı bellekte | **Expert'ler diskten akıtılır** |
| LoRA | edge0'ın Recover-LoRA adaptörleri | aynı |

**Expert streaming**, 35B'nin telefonda çalışabilmesinin tek yolu: 23 GB'lık
kontrol noktası hiçbir iPhone'un belleğine sığmaz, ama bir token yalnızca 4
expert'e yönlendirilir — katman başına ~6 MB. Uygulama stacked expert
tensörlerini diskte `mmap`'li tutar, yalnızca seçilen expert'leri küçük bir
slot tamponuna kopyalar ve MLX'in `gatherQuantizedMM` çekirdeğini çalıştırır.
Bu, yerleşik yolun yaptığı matematiğin aynısıdır.

## Uygulama

- **Sohbet** — token token akan yanıt; Markdown olarak render edilir (başlık,
  liste, alıntı, yatay kaydırılabilen ve kendi kopyala düğmesi olan kod
  blokları). Modelin düşünce zinciri (`<think>…`) ayrı, katlanabilir bir
  bölümde durur; model düşünürken nabız gibi atar. Yanıt başına ölçümler
  (tok/s, ilk token süresi, token sayısı, tepe bellek), kopyala/paylaş,
  durdurma.
- **Geçmiş** — sohbetler otomatik kaydedilir ve listeden geri açılır. Geri
  açmak yalnızca metni geri getirmez: oturum, konuşmanın geçmişiyle yeniden
  kurulur, yani model kaldığı yerden **hatırlayarak** devam eder.
- **Modeller** — her tier için boyut, beklenen tepe bellek, diskteki yer ve boş
  alan; canlı ilerlemeli indirme, iptal ve silme.
- **Ayarlar** — sıcaklık / top-p / top-k / tekrar cezası / maks. token (tier
  varsayılanlarıyla), sistem istemi, düşünme modu, LoRA anahtarı, MLX önbellek
  sınırı, expert önbellek bütçesi ve canlı bellek/önbellek göstergeleri.
- **Tanılama** — Ayarlar'ın altındaki *Tanılama bilgisini kopyala* düğmesi;
  yüklü tier, sağlık kontrolü sonucu ve modelin açılışta ürettiği örnek,
  eşleşen/eşleşmeyen LoRA hedefleri, üretim parametreleri, expert önbellek
  isabeti ve bellek rakamlarını tek blok halinde panoya kopyalar. Model saçma
  çıktı verirse bildirilecek şey budur.

Model dosyaları `Application Support` altında tutulur (iOS'un temizleyebildiği
`Caches` değil) ve iCloud yedeğinden hariç tutulur. Sohbetler de aynı yerde,
sohbet başına bir JSON dosyası olarak saklanır.

### Hazır indirilmiş modeli kullanmak

35B ~23 GB; dosyalar zaten bir Mac'te, iCloud Drive'da veya USB-C diskte
duruyorsa telefondan tekrar indirmeye gerek yok. Modeller sekmesindeki
**klasör** düğmesi bir klasör seçtirir ve gerekli dosyaları uygulamanın kendi
deposuna kopyalar.

Klasörde şunlar olmalı:

| Dosya | Gerekli mi |
|---|---|
| `config.json` | **Evet** — mimari ve quantization bilgisi |
| `model-*.safetensors` (+ `model.safetensors.index.json`) | **Evet** |
| `tokenizer.json`, `tokenizer_config.json` | **Evet** |
| `generation_config.json`, chat template (`*.jinja`) | varsa kopyalanır |
| `lora_edge0_35b.safetensors` | **Evet** — Recover-LoRA; olmazsa kalite düşer |
| `prerouter_edge0_35b.safetensors` | **Hayır** — bu uygulama prerouter'ı kullanmıyor, kopyalanmaz |

### Bellek

35B'nin expert önbelleği **MB cinsinden** ayarlanır, katman başına expert
sayısıyla değil: bütçe tüm MoE katmanları arasında paylaşılır ve yükleyici onu
kontrol noktasının gerçek expert boyutuna bölerek katman başına slot sayısını
bulur. Varsayılan, cihaz belleğinin on altıda biridir.

iOS bellek uyarısı gönderdiğinde uygulamanın geri vermek için birkaç saniyesi
vardır; o anda expert önbellekleri boşaltılır. Model yüklü kalır — sonraki adım
o ağırlıkları diskten yeniden okur, yani yanıt değişmez, yalnızca yavaşlar.

## IPA'yı almak

Her push'ta GitHub Actions bir macOS runner'da **imzasız IPA** üretir:

1. [Actions sekmesi](../../actions/workflows/build-unsigned-ipa.yml) → en son
   çalıştırma
2. Sayfanın altındaki **Artifacts** bölümünden `Edge0Demo-unsigned-ipa` indir
3. Zip'ten çıkan `.ipa`'yı **Feather** ile kendi sertifikanla imzala ve kur

Yerelde derlemek için (Mac + Xcode gerekir):

```bash
brew install xcodegen
cd ios && xcodegen generate && open Edge0Demo.xcodeproj
```

## Bilinen sınırlar

- Modellerin çıktı kalitesi cihazda **henüz doğrulanmadı**. 8B mimarisi bu
  depoda elle portlandı; sayısal bir hata çıktıyı bozabilir. Yükleme sonrası
  sağlık kontrolü 12 adımlık bir üretim yapıp sonucu Ayarlar'da gösterir —
  oradaki "açılış örneği" anlamsızsa sorun porttadır, sohbeti denemeye gerek
  kalmadan bellidir.
- 35B'de uzun promptlar yavaştır: prompt işlenirken çok sayıda farklı expert
  diskten okunur. Kısa promptlarda ve devam eden sohbette çok daha hızlıdır
  (yönlendirme ardışık tokenlar arasında büyük ölçüde aynı kalır).
- 23 GB'lık indirme Wi-Fi ve uygulamanın açık kalmasını gerektirir.
- 35B, imzalama sırasında **artırılmış bellek sınırı** yetkisinden
  (`com.apple.developer.kernel.increased-memory-limit`) faydalanır. Feather ile
  imzalarken bu yetki profilinde varsa açık bırak; yoksa 35B jetsam sınırına
  daha erken takılabilir — bu durumda Ayarlar'dan expert önbellek bütçesini
  düşür.

## Kaynaklar ve lisans

- Modeller, adaptörler ve streaming fikri: [Edge0-AI/edge0](https://github.com/Edge0-AI/edge0) (Apache-2.0)
- Çalışma zamanı: [mlx-swift](https://github.com/ml-explore/mlx-swift) ve
  [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) (MIT)
- `ios/Edge0Demo/Edge0Model/Vendor/` altındaki dosyalar mlx-swift-lm'den
  kopyalanmıştır; neden kopyalandıkları dosya başlıklarında açıklanmıştır.
