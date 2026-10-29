# 初回導入構成と出典

2026-10-26。新規Windows導入では、明示した計画・導入操作で対応する最新安定ソフトウェアを解決します。解決した版・取得元・容量・SHA-256・日時を導入先の状態へ保存し、中断後は同じ構成で再開します。`recipe.json` は対象部品と導入条件の基礎定義です。古い固定archiveへ黙って戻すためのfallbackには使いません。

重み、配布元archive、installer、DLL、第三者wheelはこのリポジトリに同梱しません。取得したarchive内のLICENSEとNOTICEは導入先へ保持します。既存ComfyUIの本体・GPU依存と利用者の変更は自動更新・上書きしません。

| 部品 | 出典と条件 | このアプリでの扱い |
| --- | --- | --- |
| ComfyUI | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)・GPL-3.0 | 最新安定releaseのtagをcommitへ解決し、archiveの容量・hashを照合。旧固定core用の互換patchを流用しない |
| H3 Motion Context | [seitanism](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)・GPL-3.0 | 空間noise maskとV2V参照node。許可した配布元の現行commitへ解決 |
| VideoHelperSuite | [Kosinkadink](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)・GPL-3.0 | 参照動画の入力node。取得commit・archive hashを記録 |
| FFmpeg / ffprobe | [Gyan build](https://www.gyan.dev/ffmpeg/builds/)・GPL-3.0を含む外部Windows build | 対応する最新安定buildを解決して別置き。LICENSEとsource案内を保持。PATH設定はアプリprocess内だけ |
| Python | [Python公式](https://www.python.org/about/legal/)・PSF等 | 対応系列の公式x64 installer。署名・容量・hashを検証し、必要時にユーザーフォルダーへ導入 |
| VC runtime | [Microsoft公式](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)・Microsoftの条件 | 不足時に公式署名付きinstallerを取得。NVIDIAドライバーやRust構築用C++ Build Tools / SDKとは別 |

同梱の `h3_trial_chunked_mlp.py` は公開Torch / ComfyUI APIによるFFN分割、`mosaic_h3_memory.py` は公開model_management APIによる段間解放、`mosaic_h3_public.py` は動画latentの空間軸をbicubicで縦横2倍にする独自nodeです。新しい公開用拡大nodeは追加重み・LoRAを使わず、audio latentを維持します。

旧 `h3_guided_latent_upscaler.py` とH3 Video Upsamplerの固定版に関する記録は互換・検収の履歴です。公開用profileの導入・生成では使いません。旧外部nodeは[元配布元](https://github.com/dntpi/ComfyUI-H3-Video-Upsampler)がMITと第三者告知を持ち、重みの条件とは別です。

GPU wheelは公式PyTorch CUDA 13.0 indexとPyPIから取得します。最新安定Torchと対応するtorchvision / torchaudio / Windows用Tritonをmetadataから選び、wheelとmetadataのhashを記録します。Torchは修正確認済みの下限2.14.1以上です。comfy-kitchen / comfy-aimdoは選択したComfyUIのrequirementsに従います。対応wheelや配布元情報を確認できなければ停止します。合成GPU検査の成功は、実H3完走・画質の証明とは区別します。

ベース・補助資源はH3由来INT4 DiT / encoder、FP16 Video VAE、FP32 Audio VAEと、BasicVSR++ / RF-DETRです。ソフトウェアの最新版解決で別hashへ置換しません。2026-10-27にUnlocked / Decensoreの主LoRA一つを必須とする要求へ更新しましたが、条件の確認待ちでモデル導入・自動取得・実推論は停止中です。Eros・guided重み・学習済みupscalerは戻しません。[主LoRAの出典と残件](../../docs/required-lora.md)、[導入手順](../../docs/model-setup.md)、[利用条件](../../docs/use-conditions.md)を参照してください。

独自部分はルート[LICENSE](../../LICENSE)のAGPL-3.0-onlyです。H3と各モデルの条件は別に適用され、地域・AUP・商用・製品提供側の安全措置等の確認を、重み非同梱だけで済ませたとは扱いません。重みを含むinstallerや外部推論サービスは本候補の提供範囲に含みません。
