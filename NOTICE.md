# mosaic-refiner の表示と利用許諾

Copyright (c) 2026 Studio Yebisu and mosaic-refiner contributors.

開発リポジトリ: [studio-yebisu/mosaic-refiner](https://github.com/studio-yebisu/mosaic-refiner)

本リポジトリの独自コード・文書は、別の表示があるファイルを除き **GNU Affero General Public License version 3 only (`AGPL-3.0-only`)** で提供します。[LICENSE](LICENSE)の全文を参照してください。無保証です。同梱する第三者コードの著作権表示とライセンスは維持します。

Lada / OpenMMLab由来のBasicVSR++推論部品を含むため、アプリの独自部分にもAGPL v3を選びました。各ファイルのApache / AGPL表示をアプリの表示で上書きしません。[第三者部品一覧](THIRD_PARTY_NOTICES.md)、[vendor NOTICE](mosaic_refiner/vendor/lada_basicvsr/NOTICE.md)と同梱本文を参照してください。MosaicDiff本体のソースは取り込んでいません。

アプリの取得・改変・配布・商用利用はAGPLの条件に従って行えます。第三者モデル、外部ComfyUI、node、FFmpeg、Python、ライブラリの利用権はこの表示から付与しません。モデル・素材・実行環境のarchive・第三者wheel・構築済みbinaryは同梱しません。

H3は外部モデルです。ベースには元H3条件を明示するコミュニティ量子化・repack版を使います。MiniMax本人の直配布ファイルという意味ではありません。Unlocked V2 / DecensoreH3の条件確認待ちでモデル導入・推論は停止中です。[LoRA候補と確認範囲](docs/required-lora.md)を参照してください。公式条件の照合用コピーを[licenses/MiniMax-H3-LICENSE.txt](licenses/MiniMax-H3-LICENSE.txt)に保持します。現行の配布元条件、対象地域・AUP・商用・製品提供の条件も確認してください。このコピーはアプリ全体のライセンスではありません。

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

本アプリはMiniMax・Lada・MosaicDiff各開発元の公式製品ではありません。外部モデルによるAI生成は、元情報の復元・事実の証明を保証しません。利用の入口は[README](README.md)、用途とモデル別の条件は[利用条件](docs/use-conditions.md)です。
