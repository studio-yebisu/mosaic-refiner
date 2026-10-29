# コードとモデルの利用条件

独自コードは[AGPL-3.0-only](../LICENSE)です。[NOTICE](../NOTICE.md)、[第三者部品一覧](../THIRD_PARTY_NOTICES.md)、同梱本文を保持してください。コードの公開は外部モデル・第三者プログラムの利用権や生成物の権利を付与しません。

権利を持つ動画か必要な加工許可を得た動画を使い、人物の同意・プライバシー・肖像等の権利を確認してください。出力はAIによる推測で、元情報の回復や事実の証明を保証しません。

## 外部モデル

H3には[公式Community License / Exhibit A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE)を適用し、[照合用本文](../licenses/MiniMax-H3-LICENSE.txt)も保持します。地域、AUP、商用、配布・下流提供、安全措置等は本文を確認してください。ローカル・無料・量子化・重み非同梱を理由に例外としません。配布元の現行条件も確認する必要があります。

| モデル | 出典・条件の参照先 |
| --- | --- |
| H3由来INT4 DiT / encoder | [Merserkの固定card](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot/blob/3c167bc916cab6cf3b85b9e3769952757ebb2e5d/README.md)。H3由来community conversionとして扱う |
| H3 Video / Audio VAE | [Comfy-Orgの固定card](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/README.md)。H3由来repackとして扱う |
| BasicVSR++ | [Ladaの固定配布](https://huggingface.co/ladaapp/lada/tree/bcf461d46d9a98981fc64b815df5178f42215cdf)と[対応条件](https://github.com/ladaapp/lada/blob/20cb34a20a83c72c87a991d2c949032c70085b16/LICENSE.md) |
| RF-DETR | [Jasna作者の条件](https://github.com/Kruk2/jasna/blob/79d9f11582c03accc16197896780241c54ba2823/LICENSING.md)。作者の専用detector条件として扱う |
| 必須主LoRA | [作者・固定版・未解決範囲](required-lora.md)を参照。現在の導入・推論は停止 |

コミュニティ変換版をMiniMax本人の直配布とは表記しません。固定hashはファイルの同一性の検査であり、用途や地域の適合を証明しません。本人の認証や利用条件のチェックも作者承認とは別です。

提供対象はソースと必要文書です。モデル、素材、テスト用ファイル、内部検証記録、過去の開発履歴、構築済みbinary、第三者wheel、モデル込みinstaller、外部推論サービスは提供しません。[実装状況](implementation-plan.md)に未確認を記載しています。
