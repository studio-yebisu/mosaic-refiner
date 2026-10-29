"""Planning, processing and explicit setup CLI; import has no model side effects."""

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from . import __version__
from .checkpoint import request_cancel, summarize
from .geometry import Rect
from .h3_adapter import inspect_runtime, prepare_request
from .h3_inputs import materialize_window
from .media import probe
from .pipeline import run
from .planning import model_windows, plan_cfr


def positive_fraction(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("正の整数、小数、または分数を指定してください") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("正数を指定してください")
    return result


def nonnegative_fraction(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("0以上の数または分数を指定してください") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("0以上を指定してください")
    return result


def rectangle(value: str) -> Rect:
    try:
        coordinates = [int(part) for part in value.split(",")]
        if len(coordinates) != 4:
            raise ValueError
        return Rect(*coordinates)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROIはx,y,width,heightの整数4つです") from exc


def build_parser():
    parser = argparse.ArgumentParser(description="領域動画補正・取消し/再開・Windows GUI")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("gui", help="動画drop/選択 → 開始のWindowsウィンドウ")
    web = commands.add_parser("web", help="ローカルWebGUI（動画選択・開始・進捗・取消し・再開）")
    web.add_argument("--port", type=int, default=8765)
    commands.add_parser("setup-gui", help="PC要件・既存/新規ComfyUI・モデル/依存の導入画面")
    setup = commands.add_parser("setup-runtime", help="PC要件と初回導入。起動だけでは取得しません")
    setup.add_argument("action", choices=("configure", "check", "models-download", "node-install",
                       "libraries-plan", "libraries-install", "import-unlocked", "plan", "install", "start-managed"))
    setup.add_argument("--root", type=Path, help="既存ComfyUIのmain.pyがあるroot")
    setup.add_argument("--python", type=Path, help="このComfyUIを起動するPython")
    setup.add_argument("--endpoint")
    setup.add_argument("--new", action="store_true", help="空のrootへ新規導入（既存ComfyUI使用中は拒否）")
    setup.add_argument("--acknowledge-warnings", action="store_true", help="12〜23GB VRAM等の未検証条件を確認済み")
    setup.add_argument("--accept-model-terms", action="store_true")
    setup.add_argument("--lora-file", type=Path)
    setup.add_argument("--lora-profile", choices=("unlocked_v2", "restoration"), default="unlocked_v2",
                       help="必須LoRAを選択。取得には本人のCivitai API認証が必要")
    models = commands.add_parser("models", help="不足モデルの確認・配布元から取得（明示操作のみ）")
    models.add_argument("action", choices=("check", "download"))
    models.add_argument("--models-dir", type=Path, required=True)
    models.add_argument("--cache-dir", type=Path)
    models.add_argument("--verify", action="store_true", help="既存モデルの全bytes hashを検査")
    models.add_argument("--accept-model-terms", action="store_true", help="台帳の配布元利用条件を確認済み")
    models.add_argument("--catalog", type=Path, help="本人が条件を確認した固定版JSON台帳（Hugging Face / Civitai）")
    auth = commands.add_parser("auth", help="本人のモデル取得認証。キーを引数やJSONへ保存しません")
    auth.add_argument("action", choices=("login", "check", "forget"))
    auth.add_argument("provider", choices=("huggingface", "civitai"))
    auth.add_argument("--save", action="store_true", help="Windows資格情報マネージャーへ保存")
    refine = commands.add_parser("refine", help="実H3で領域動画を処理。ROI省略はブロック候補検出")
    refine.add_argument("input", type=Path)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--job-dir", type=Path, required=True)
    refine.add_argument("--roi", type=rectangle)
    refine.add_argument("--context-seconds", type=int, choices=(5, 10, 15), default=5)
    refine.add_argument("--unit-seconds", type=positive_fraction, default=Fraction(15))
    refine.add_argument("--mode", choices=("speed", "balanced", "detail"), default="balanced",
                        help="速度優先 / 標準 / 精細さ優先（画質は未評価）")
    refine.add_argument("--render-scale", type=int, choices=(50, 75, 100, 125))
    refine.add_argument("--pixel-budget", type=int, choices=(65536, 147456, 262144, 589824))
    refine.add_argument("--crop-margin", type=int, choices=(0, 16, 32, 64))
    refine.add_argument("--sampling", choices=("light", "standard"), help="軽め12+6 / 標準20+10")
    refine.add_argument("--h3-lora", choices=("unlocked_v2", "restoration"), default="unlocked_v2",
                        help="追加LoRA必須: Unlocked V2 / DecensoreH3（許諾確認待ちのため実行停止中）")
    refine.add_argument("--lora-strength", type=float, default=1.0, help="必須LoRAの強度。0より大きく2以下")
    plan = commands.add_parser("plan", help="CFRの入力フレームを保存単位へ割り当てる")
    plan.add_argument("--frames", type=int, required=True, help="元動画のフレーム数")
    plan.add_argument("--fps", type=positive_fraction, required=True, help="元FPS。30000/1001等も指定可")
    plan.add_argument("--unit-seconds", type=positive_fraction, default=Fraction(15))
    plan.add_argument("--context-seconds", type=int, choices=(5, 10, 15), default=5)
    plan.add_argument("--with-windows", action="store_true", help="未検証H3候補窓とpadding計画も表示")
    plan.add_argument("--overlap-seconds", type=nonnegative_fraction, default=Fraction(1))
    plan.add_argument("--scene-cut-frame", type=int, action="append", default=[])
    media_probe = commands.add_parser("probe", help="実動画を走査し、CPU版の対応形式と原PTSを確認")
    media_probe.add_argument("input", type=Path)
    execute = commands.add_parser("run", help="CPU診断処理。identity=無加工、invert=ROI反転。AI補正ではありません")
    execute.add_argument("input", type=Path)
    execute.add_argument("--output", type=Path, required=True, help=".nutまたはlossless RGBの.mp4。既存ファイルを上書きしません")
    execute.add_argument("--job-dir", type=Path, required=True, help="checkpoint保存先。同じ引数で再実行すると再開")
    execute.add_argument("--roi", type=rectangle, required=True)
    execute.add_argument("--backend", choices=("identity", "invert"), default="identity")
    execute.add_argument("--feather", type=int, default=0)
    execute.add_argument("--first-frame", type=int, default=0)
    execute.add_argument("--end-frame", type=int, help="対象区間の末尾（このframeを含まない）")
    execute.add_argument("--unit-seconds", type=positive_fraction, default=Fraction(15))
    execute.add_argument("--context-seconds", type=int, choices=(5, 10, 15), default=5,
                         help="将来H3用。CPU処理には適用せず、設定hashに保存")
    execute.add_argument("--stop-after-units", type=int, help="指定数の新規chunk確定後に安全停止（再開試験用）")
    for name, help_text in (("status", "job状態を表示（artifact再hashはrun再開時）"),
                            ("cancel", "自分のCPU jobへ取消し要求を保存")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--job-dir", type=Path, required=True)
    h3 = commands.add_parser("h3-plan", help="領域動画/maskの接続計画JSON。WF生成/送信はしません")
    h3.add_argument("input", type=Path)
    h3.add_argument("--roi", type=rectangle, required=True)
    h3.add_argument("--context-seconds", type=int, choices=(5, 10, 15), default=5)
    h3.add_argument("--unit-seconds", type=positive_fraction, default=Fraction(15))
    h3.add_argument("--overlap-seconds", type=nonnegative_fraction, default=Fraction(1))
    h3.add_argument("--scene-cut-frame", type=int, action="append", default=[])
    h3.add_argument("--max-pixels", type=int, default=262144)
    h3.add_argument("--seed", type=int, default=0)
    h3.add_argument("--prompt-file", type=Path)
    h3.add_argument("--crop-margin", type=int, default=0, help="書戻しROI外に含める参照用余白pixel")
    h3.add_argument("--feather", type=int, default=0, help="書戻しROI内側のfeather pixel")
    prepare = commands.add_parser("h3-prepare", parents=[h3], add_help=False,
                                  help="1窓の参照動画/mask/時刻表をCPU保存。AI推論・WF送信なし")
    prepare.add_argument("--output-dir", type=Path, required=True, help="新しいbundle保存先。既存directoryは拒否")
    prepare.add_argument("--window-index", type=int, default=0, help="h3-planで確認した窓番号。既定0")
    commands.add_parser("h3-check", help="選択したComfyUIのqueue/公開APIを読取。起動/再起動/送信なし")
    return parser


def main(argv: list[str] | None = None) -> int:
    import os
    import sys
    arguments = sys.argv[1:] if argv is None else argv
    if not os.environ.get("MOSAIC_REFINER_PYTHON_WORKER") and arguments[:1] != ["auth"]:
        from .native_host import invoke_if_available
        native_result = invoke_if_available(sys.argv[1:] if argv is None else argv)
        if native_result is not None:
            return native_result
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "auth":
            from .model_auth import Credentials, WindowsVault, check as check_auth
            vault=WindowsVault()
            if args.action=="forget":
                vault.delete(args.provider)
                result={"provider":args.provider,"deleted":True}
            elif args.action=="login":
                if not sys.stdin.isatty():
                    raise ValueError("キーは本人の対話端末で入力してください。AIの実行ログやpipeから受け取りません")
                import getpass
                credentials=Credentials()
                credentials.set(args.provider,getpass.getpass("本人のAPI認証情報（画面に表示しません）: "))
                result=check_auth(args.provider,credentials)
                if args.save:vault.save(args.provider,credentials.tokens[args.provider])
                result["saved"]=args.save
                if not args.save:result["next"]="次回取得で使うには--saveを指定するか、セットアップ画面で入力してください"
            else:result=check_auth(args.provider,Credentials.saved(vault))
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0
        if args.command == "web":
            from .web import main as web_main
            web_main(args.port)
            return 0
        if args.command == "setup-gui":
            from .setup_gui import main as setup_main
            setup_main()
            return 0
        if args.command == "setup-runtime":
            from .runtime_settings import load, save, RuntimeConnection
            from .setup_runtime import (inspect_nodes, resource_status, acquire_resources,
                                        install_reference_node, prepare_libraries, import_unlocked)
            from .setup_environment import plan as setup_plan, install_all, start_managed, install_nodes
            if args.root:
                python=args.python or args.root/'.venv/Scripts/python.exe'
                connection=RuntimeConnection(args.root,python,args.endpoint or 'http://127.0.0.1:8190',True)
            else:
                if args.python or args.endpoint or args.new:
                    raise ValueError("--python/--endpoint/--newには--rootも指定してください")
                connection=load()
            if args.action == "plan":
                result=setup_plan(connection,args.new,lora_profile=args.lora_profile)
            elif args.action == "install":
                result=install_all(connection,args.new,args.accept_model_terms,args.acknowledge_warnings,args.lora_file,lora_profile=args.lora_profile)
            elif args.action == "start-managed":
                result=start_managed(connection.validate())
            elif args.action == "configure":
                if not args.root or not args.python:
                    raise ValueError("configureには--rootと--pythonが必要です")
                result = {"settings": str(save(connection)), "restart_application_required": True}
            else:
                connection.validate()
                if args.action == "check":
                    result = {"connection": connection.to_dict(), "nodes": inspect_nodes(connection),
                              "models": resource_status(connection, verify=False,lora_profile=args.lora_profile), "runtime_ready": False}
                elif args.action == "models-download":
                    if not args.accept_model_terms:
                        raise ValueError("各配布元の利用条件を確認し、--accept-model-termsを指定してください")
                    result = {"models": acquire_resources(connection,lora_profile=args.lora_profile,accept_terms=args.accept_model_terms), "runtime_ready": False}
                elif args.action == "node-install":
                    if not args.accept_model_terms:raise ValueError('ノードの配布条件を確認し、--accept-model-termsを指定してください')
                    result = install_nodes(connection)
                elif args.action == "import-unlocked":
                    if not args.lora_file:
                        raise ValueError("--lora-fileにUnlocked V2を指定してください")
                    result = import_unlocked(connection, args.lora_file)
                else:
                    result = prepare_libraries(connection, apply=args.action == "libraries-install")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        elif args.command == "models":
            from .model_setup import inspect, install, catalog
            selected=catalog(args.catalog)
            if args.action == "download":
                if not args.accept_model_terms:
                    raise ValueError("models checkで配布元/利用条件を確認し、--accept-model-termsを指定してください")
                result = install(args.models_dir, args.cache_dir,models=selected,terms_confirmed=True)
            else:
                result = {"models": inspect(args.models_dir,models=selected,verify=args.verify), "runtime_ready": False}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "gui":
            from .gui import main as gui_main
            gui_main()
            return 0
        elif args.command == "refine":
            from .application import ApplicationJob
            options = {"mode": args.mode, "lora_profile": args.h3_lora, "lora_strength": args.lora_strength}
            for key, value in (("render_scale_percent", args.render_scale), ("max_pixels", args.pixel_budget),
                               ("crop_margin", args.crop_margin), ("sampling", args.sampling)):
                if value is not None:
                    options[key] = value
            result = ApplicationJob(args.input, args.output, args.job_dir, args.roi,
                                    context_seconds=args.context_seconds, unit_seconds=args.unit_seconds,
                                    processing_options=options).execute()
        elif args.command == "plan":
            if args.scene_cut_frame and not args.with_windows:
                parser.error("scene境界は--with-windowsと併用してください")
            result = plan_cfr(args.frames, args.fps, args.unit_seconds, args.context_seconds).to_dict()
            if args.with_windows:
                result["model_windows"] = [asdict(w) for w in model_windows(
                    args.frames, args.fps, args.context_seconds, args.overlap_seconds, tuple(args.scene_cut_frame))]
        elif args.command == "probe":
            result = probe(args.input).to_dict()
        elif args.command == "run":
            result = run(args.input, args.output, args.job_dir, args.roi, args.backend, args.feather,
                         args.unit_seconds, args.context_seconds, args.first_frame, args.end_frame,
                         args.stop_after_units)
        elif args.command in ("h3-plan", "h3-prepare"):
            prompt = args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else "Preserve the scene and motion in <Video 1>."
            result = prepare_request(args.input, args.roi, args.context_seconds, args.unit_seconds,
                                     args.overlap_seconds, tuple(args.scene_cut_frame), args.max_pixels, args.seed, prompt,
                                     args.crop_margin, args.feather)
            if args.command == "h3-prepare":
                result = materialize_window(args.input, result, args.output_dir, args.window_index)
        elif args.command == "h3-check":
            result = inspect_runtime()
        elif args.command == "cancel":
            request_cancel(args.job_dir)
            result = {"cancel_requested": True}
        else:
            result = summarize(args.job_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"エラー: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "取消しました。確定済みcheckpointは保持しています。\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 130 if result.get("state") == "cancelled" else 0


if __name__ == "__main__":
    raise SystemExit(main())
