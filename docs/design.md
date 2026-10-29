# 設計

Rust hostがCLI・ローカルHTTP・ジョブ所有・進捗・取消し・再開を管理します。Python workerが動画処理、AI接続、セットアップを担当し、Rust coreがRGB合成と数値処理を行います。通常画面はloopbackのWeb画面で、ComfyUIのAPIとは別の操作サーバーです。

処理経路は、原PTSとmetadataの取得 → 検出・追跡・形状mask → cropと生成窓 → BasicVSR++ → H3低段 → 重み不要のbicubic latent x2 → H3高段 → 原座標への合成 → 音声・時刻・対象外画素の検査です。選択した主LoRAを両H3段の共通モデルへ適用します。実行前にモデルとLoRAの固定ファイルを照合し、不足・不一致なら取得方法を案内して停止します。

ジョブごとにOS排他lockを持ち、partialから検証後に確定し、入力・設定・実装・モデル・runtimeのhashを再開時に照合します。異なる旧profileへ自動置換して再開しません。取消しで共有ComfyUI全体をinterruptせず、自分の安全な保存点まで待ちます。

原本・既存出力を上書きせず、完成出力の全編decode、PTS、末尾、音声、対象外RGB等を検査します。プレビューは明示操作時だけ表示します。完成後の中間素材削除も所有確認と利用者の明示操作を必要とします。

取得は本人認証、固定model / version / file ID・名前・容量・SHA、公式配布元、URL / redirectの制限を使います。キーをジョブ・通常JSON・ログへ保存しません。コードの条件と外部重みの条件は分けます。[モデル準備](model-setup.md)と[セキュリティ](security.md)を参照してください。
