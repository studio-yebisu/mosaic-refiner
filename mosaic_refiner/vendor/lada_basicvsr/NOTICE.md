# Lada BasicVSR++ 推論部品

取得元: https://github.com/ladaapp/lada/tree/20cb34a20a83c72c87a991d2c949032c70085b16/lada/models/basicvsrpp

原著者: Lada Authors、OpenMMLab。network.py / flow_warp.py は Apache-2.0 AND AGPL-3.0、deformconv.py は AGPL-3.0。ライセンス全文を同梱する。

2026-10-09変更: 推論に不要なMMEngine registry / checkpoint helperを外し、同梱support.pyのTorch初期化関数を使う。数式・network forwardは原実装を維持する。完全な重みをstrict loadし、学習・checkpoint自動取得はしない。MosaicDiffのソースは使用していない。

アプリの独自部分にもルートLICENSEのAGPL-3.0-onlyを適用する。各第三者ファイルのApache / AGPL表示と全文を維持し、ルートのTHIRD_PARTY_NOTICES.mdから参照する。モデル重みと素材は同梱しない。
