# 第三者コード・外部依存・モデル

2026-10-27。提供対象はアプリのソースのみです。同梱するコードと、利用者の明示操作で別途取得するものを分けます。アプリのAGPL表示を、重みや外部プログラムの許諾へ読み替えません。H3由来ベースにUnlocked V2 / DecensoreH3の一つを必須とし、本人のCivitai API認証で取得して使います。[現行の確認範囲](docs/required-lora.md)を参照してください。

## 同梱するコード

| 場所 | 著作者・出典・固定版 | 条件と変更 |
| --- | --- | --- |
| `mosaic_refiner/vendor/lada_basicvsr/network.py`・`flow_warp.py` | OpenMMLab / Lada Authors、[Lada](https://github.com/ladaapp/lada/tree/20cb34a20a83c72c87a991d2c949032c70085b16/lada/models/basicvsrpp)、`20cb34a20a83c72c87a991d2c949032c70085b16` | Apache-2.0 AND AGPL-3.0。MMEngine registry等を外し、Torch推論supportを追加。数学・forwardを維持 |
| `mosaic_refiner/vendor/lada_basicvsr/deformconv.py` | Lada Authors、同固定版 | AGPL-3.0。Torch推論部品 |

[vendor NOTICE](mosaic_refiner/vendor/lada_basicvsr/NOTICE.md)、[AGPL本文](mosaic_refiner/vendor/lada_basicvsr/AGPL-3.0.md)、[Apache本文](mosaic_refiner/vendor/lada_basicvsr/Apache-2.0.txt)、[GPL本文](licenses/GPL-3.0.txt)を保持します。support / adapter / Rust core / hostと、それ以外の独自実装はルート[NOTICE](NOTICE.md)・[LICENSE](LICENSE)に従います。

旧recipeの互換patchは[過去のComfyUI固定版](https://github.com/Comfy-Org/ComfyUI/tree/f4d76984586901b53b424205eb666d5a4563deb2)の短い原文を含んでいました。元本体はGPL-3.0です。現行の最新安定版取得にその旧patchを流用しません。過去の検収・履歴の表示も当時のものとして保持します。

## 別途取得するコード・実行環境

| 対象 | 出典・主な表示条件 | 配布形態 |
| --- | --- | --- |
| ComfyUI・Motion Context・VideoHelperSuite | GPL-3.0、[導入NOTICE](mosaic_refiner/setup_assets/NOTICE.md) | 明示操作で選択したrelease / commitのarchiveを取得し、LICENSE / NOTICEと取得hashを保持 |
| FFmpeg / ffprobe | GPL-3.0等、[Gyan build](https://www.gyan.dev/ffmpeg/builds/) | 対応する最新安定Windows buildを解決して別途取得。配布元のLICENSEとsource案内を保持。自作配布物へbinaryを同梱しない |
| Python | PSF等、[公式条件](https://www.python.org/about/legal/) | 公式署名付きinstallerを外部取得 |
| Microsoft VC runtime / C++ Build Tools | [Microsoftの条件](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/) | runtimeは不足時の明示導入、Build Tools / SDKは利用者が準備 |
| Torch等のGPU依存 | 各配布元の本文・第三者告知、[導入NOTICE](mosaic_refiner/setup_assets/NOTICE.md) | 明示操作で外部wheelを取得。アプリのGit / wheelへ同梱しない |

アプリ側で取得するNumPy、PyAV、psutil、websocket-client、OpenCV、ONNX Runtimeにも各配布元の条件が適用されます。NumPyはBSD等、PyAVはBSD-3-Clause、psutilはBSD-3-Clause、websocket-clientはApache-2.0、OpenCVはApache-2.0等、ONNX RuntimeはMIT等の表示です。各wheel内のFFmpeg・OpenBLASその他の第三者部品は別表示を持ちます。pipが取得したwheelを、この一覧だけを根拠に再配布しないでください。

新規導入は最新安定ComfyUIと対応するGPU依存を解決し、版・URL・容量・hash・解決日時を状態へ保存します。再開はその構成を使います。既存の共有環境は自動更新しません。モデルはソフトウェアの最新版に合わせて置換せず、許諾を照合した固定fileを使います。旧guided経路で利用したH3 Video Upsamplerは[元配布元](https://github.com/dntpi/ComfyUI-H3-Video-Upsampler)がMITと第三者告知を持ちますが、公開用profileの必須部品からは除外しました。

Rust hostのWindows用固定依存49件の名前・版・license・repositoryは[依存台帳](docs/rust-host-dependencies.json)にあります。公開候補にcrate source / binaryは同梱しません。将来exeを配布する場合は、その構築物に含まれる依存の本文・NOTICE・対応source等を改めて揃えます。

## モデルと再配布

モデル・LoRA・素材は同梱しません。公開用のH3はMerserkのINT4 DiT / encoder、Comfy-OrgのFP16 Video VAE / FP32 Audio VAEで、元H3条件を明示するコミュニティ量子化・repack版です。補助モデルはLada BasicVSR++とJasna RF-DETRです。固定URL・版・容量・hashは[H3台帳](mosaic_refiner/model_catalog.json)と補助モデルの実装が正本です。[モデル別条件](docs/use-conditions.md)を参照してください。

H3の地域・AUP・商用・提供側の条件は量子化版にも適用されます。hash一致、重み非同梱、LoRAの有無を、用途の許可や提供側の安全措置等の確認に代えません。現行profileでは旧Eros / guided重み等へ自動代用せず、必須主LoRAと固定モデルを照合して補正を実行します。

必須主LoRA候補の著作者表示を以下に保持します。どちらの重みも同梱していません。この一覧は追加の利用許諾ではありません。

| 主LoRA候補 | 著作者・固定配布ページ | 適用条件の確認状況 |
| --- | --- | --- |
| MiniMax H3 Unlocked V2 | [The_Midnight_Lab](https://civitai.com/user/The_Midnight_Lab) / [model 2925727・version 3354822](https://civitai.com/models/2925727?modelVersionId=3354822) | 作者はcreditを要求。作者とH3の条件を確認して使用 |
| DecensoreH3 | [herpderpmerp](https://civitai.com/user/herpderpmerp) / [model 2990026・version 3390424](https://civitai.com/models/2990026?modelVersionId=3390424) | 作者はcreditを要求。動画の加工・出力利用の条件を確認して使用 |

Civitai上のH3配布等の契約と取得後の自前実行は区別します。`allowDerivatives`の表示はマージ共有の意味です。SD用の個別条件生成器をH3へ転用せず、作者の許可本文を創作しません。[主LoRAの条件](docs/required-lora.md)に適用する条件を記載しています。

改変箇所の表示、著作権表示と本文の保持、AGPL対象の対応source提供等を、実際に含める部品の条件に従って行ってください。元Gitへのリンクだけで全形式の再配布条件を満たすと断定しません。本候補はソースで、重みや外部環境を含む完成版installerではありません。
