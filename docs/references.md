# 実装と依存の出典

アプリは公開APIと処理アイデアを基に独自実装します。MosaicDiff本体のソースは取り込んでいません。第三者コードの表示・本文は[一覧](../THIRD_PARTY_NOTICES.md)とvendorに保持します。

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI): 外部の推論runtime。
- [H3 Motion Context](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)、[VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite): 外部node。
- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3): 基盤の条件。モデル別の配布元は[利用条件](use-conditions.md)。
- [Lada](https://github.com/ladaapp/lada/tree/20cb34a20a83c72c87a991d2c949032c70085b16): 同梱BasicVSR++部品の出典。vendor表示を維持。
- [Jasna](https://github.com/Kruk2/jasna): 専用RF-DETRモデルの配布・条件。
- [FFmpeg build](https://www.gyan.dev/ffmpeg/builds/)、[Python](https://www.python.org/about/legal/)、[PyTorch](https://pytorch.org/): 利用者の明示操作で別途取得する外部環境。

取得した外部コードやwheelのLICENSE / NOTICEを保持し、この一覧だけで再配布条件を満たしたとは扱いません。Rust hostの外部依存は[台帳](rust-host-dependencies.json)へ記載しています。
