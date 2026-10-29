# 必須LoRAと利用条件

主LoRAは次の一つが必須です。二つを重ねず、LoRAなし・強度0・別モデルへ自動代用しません。取得には本人のCivitai API認証と利用条件の明示確認が必要です。実行時に固定ファイルの容量・SHAを照合し、両H3段へ適用して補正します。

| 候補 | 著作者・固定識別 |
| --- | --- |
| MiniMax H3 Unlocked V2 | [The_Midnight_Lab](https://civitai.com/user/The_Midnight_Lab) / [model 2925727・version 3354822](https://civitai.com/models/2925727?modelVersionId=3354822) / file 3242253 / `Minimax_H3_Unlocked_V2.safetensors` / 298,257,967 bytes |
| DecensoreH3 | [herpderpmerp](https://civitai.com/user/herpderpmerp) / [model 2990026・version 3390424](https://civitai.com/models/2990026?modelVersionId=3390424) / file 3280455 / `DecensoreH3_000002500.safetensors` / 155,110,296 bytes |

固定SHA-256:

```text
Unlocked V2  b05686a14b96bf84102274b1c355558a2f41369753b472ddc0724c6f5147503b
DecensoreH3   8c693023caa51063738cd961de1f4623ad274b003bcf6eedf677a7adbf456586
```

[実装の固定profile](../mosaic_refiner/h3_lora_profiles.py)が配置先と固定識別の正本です。Decensoreのアプリ内配置名と作者の配布ファイル名を混同しません。未知の最新版・mirrorへ切り替えません。

## 配布元の条件

Unlocked V2の作者説明は生成時の安全措置を解除する目的を述べており、H3 Exhibit A §5との整合を確認する必要があります。Decensoreの説明はモザイク補正用で、名前・LoRAという形式・NSFW表示だけから同じ解除目的や違反を断定しません。固定ファイルのローカル動画利用・取得案内・商用出力を認める条件本文と上流条件を確認します。

Civitaiの許可表示は条件本文の代わりではありません。`allowDerivatives`はマージ共有の表示であり、全派生行為の一律禁止とは読みません。Civitai上の提供に関する契約を、取得後のローカル利用の例外と推測しません。両作者のcreditを保持しますが、この表示自体が追加の許可ではありません。

本人のCivitai API認証は実際の取得で必須です。認証成功やhash一致を用途の許可そのものとは扱わず、本人が配布元の条件を確認して使います。アプリは主LoRA一つと固定ファイルの照合を必須にし、正常な構成で動画補正を実行します。実行できることと、すべての用途の許可・一般的な画質の保証は別です。[利用条件](use-conditions.md)と[導入案内](model-setup.md)を参照してください。
