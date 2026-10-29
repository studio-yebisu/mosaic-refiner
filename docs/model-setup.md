# 構築とモデル準備

**必須LoRAの条件確認待ちで、全体のモデル導入・自動取得・H3推論は停止中です。** 以下はアプリの構築・診断と取得経路の説明で、実モデル処理の成功を示しません。

## アプリを構築する

Windows x64、Rust/Cargo 1.89以上のMSVC toolchain、C++ Build Tools、Windows SDKを用意します。`setup-app.cmd`はアプリ用Pythonと依存を準備し、セットアップ画面を開きます。必要な取得はネットワークを使います。ComfyUI用Pythonをアプリ用へ転用しません。

```powershell
./.venv/Scripts/python.exe -B scripts/build_native.py
./.venv/Scripts/python.exe -B -m mosaic_refiner --help
```

CargoはPATH、`CARGO_HOME`、標準のユーザー用`.cargo`から探します。独自配置は`build_native.py --cargo <cargoの実行ファイル>`で指定できます。明示した構築では依存の取得が発生します。起動だけでRustを構築しません。

`start-web.cmd`はローカル画面、`start-setup.cmd`は導入画面を開きます。設定画面で既存ComfyUIの場所・Python・loopback APIを選び、PC要件・導入計画を確認します。既存環境の本体・GPU依存は自動置換しません。[動作要件](hardware-requirements.md)も参照してください。

## ソフトウェアと固定モデル

新規計画は公式ComfyUIの最新安定release、許可したnode repoの現行commit、互換GPU wheel、FFmpeg等を解決します。版・URL・容量・hashを記録し、中断再開で同じ構成を使います。モデルはソフトウェアの更新に合わせて置換しません。

H3ベース4資源は[固定台帳](../mosaic_refiner/model_catalog.json)、BasicVSR++は`restoration_models.py`、RF-DETRは`learned_detection.py`、必須主LoRAは[固定profile](../mosaic_refiner/h3_lora_profiles.py)が正本です。ベース・補助の6資源に主LoRA一つを加えます。取得容量はcacheと配置先の両方を見積もります。

## 本人の認証

実際のLoRA取得では本人のCivitai APIキーが必須です。「本人のAPI認証」へ本人が非表示入力し、メモリのみかWindows資格情報マネージャーへの保存を選びます。HF tokenは必要な場合だけ使います。本人の対話端末では`auth login civitai --save`を使えます。AIのチャットやCLI引数でキーを受け渡しません。

明示した取得時に本人認証と固定metadataを大容量取得前に照合し、匿名fallback・別版・mirror・自動購入を使いません。固定URLのfile ID、容量、SHAを検査し、同名の別ファイルを上書きしません。利用者の条件確認記録を作者やMiniMaxの許可記録とは扱いません。

「モデルと利用条件」の条件詳細に、選択した主LoRAの作者・固定版・配布ページ・H3条件・未確認点を表示します。LoRAを切り替えると前の条件確認チェックを解除します。本人認証やファイルの存在だけで現在の停止は解除しません。[必須LoRA](required-lora.md)を確認してください。
