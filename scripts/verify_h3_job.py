"""Nonvisual numeric verification of an owned H3 job; never render media."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaic_refiner.checkpoint import load_json, atomic_json, digest_config
from mosaic_refiner.detection import RegionSchedule
from mosaic_refiner.numeric_qa import frame_statistics
from mosaic_refiner.native_core import identity as core_identity
from mosaic_refiner.media import probe, rgb_reader, read_exact, verify_output
from mosaic_refiner.geometry import Rect


def verification_schedule(manifest, job, info):
    processor=manifest['config'].get('processor',{})
    has_digest='schedule_sha256' in processor
    expected=processor.get('schedule_sha256')
    path=job.with_name(job.name+'.analysis.json')
    analysis_file=path.is_file()
    if analysis_file:
        data=load_json(path)
    else:
        if has_digest and expected!=digest_config(None):
            raise ValueError('自動検出の解析記録がありません。手動領域へ代用せず検査を停止します')
        data=processor.get('schedule')
        if data is None:
            # Manual CLI jobs deliberately do not produce an auto-detection file.
            config=manifest['config'];roi=Rect(**config['roi'])
            roi.validate(info.width,info.height)
            first,end=config['first_frame'],config['end_frame']
            if (type(first) is not int or type(end) is not int or not 0<=first<end<=info.frame_count):
                raise ValueError('手動領域の対象frame範囲が不正です')
            data={'records':{str(first):[{'roi':config['roi']}],str(end):[]},
                  'scene_cuts':[],'max_gap_frames':end-first,'metadata':{'method':'manual'}}
    schedule=RegionSchedule({int(k):v for k,v in data['records'].items()},
                            data['scene_cuts'],data['max_gap_frames'],data.get('metadata'))
    if has_digest and (expected!=digest_config(None) or analysis_file):
        if digest_config(schedule.to_dict())!=expected:
            raise ValueError('補正領域の解析記録がmanifestのhashと一致しません')
    return schedule,data.get('metadata',{})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    manifest = load_json(args.job / "manifest.json")
    assert manifest["state"] == "complete"
    assert manifest["stage"] == "h3-region-video"
    source, output = Path(manifest["source"]), Path(manifest["output"])
    info = probe(source, allow_timeline=True)
    schedule,analysis_metadata=verification_schedule(manifest,args.job,info)
    changed = 0
    with rgb_reader(source) as old, rgb_reader(output) as new:
        for index in range(info.frame_count):
            size = info.width * info.height * 3
            a, b = read_exact(old, size), read_exact(new, size)
            stats = frame_statistics(a, b, info.width, info.height, schedule.entries_at(index),
                                     manifest["config"]["feather"])
            if not stats["outside_equal"]:
                raise ValueError(f"対象外/穴の画素が変わっています: frame {index}")
            changed += int(stats["changed"])
    windows = []
    for path in sorted(args.job.glob("windows/*/generation/result.json")):
        result = load_json(path)
        observation = load_json(path.with_name("observation.json"))
        request = load_json(path.parent.parent / "input/manifest.json")["request"]
        windows.append({"index": request["window"]["index"],
                        "canvas": request["canvas"], "window": request["window"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "node_seconds": observation["node_seconds"],
                        "device_peak_mib_including_other_processes": observation["device_peak_mib_including_other_processes"],
                        "server_peak_rss_bytes": observation["server_peak_rss_bytes"]})
    verification = verify_output(source, output, info)
    verification["ai_backend_executed"] = manifest["verification"]["ai_backend_executed"]
    verification["ai_restoration"] = manifest["verification"]["ai_restoration"]
    report = {"frames": info.frame_count, "changed_frames": changed,
              "cpu_core": core_identity(),
              "duration_seconds": float((info.frame_pts[-1] + info.frame_durations[-1] - info.frame_pts[0]) * Fraction(info.time_base)),
              "outside_rgb_exact_all_frames": True,
              "shape_exterior_and_holes_exact": bool(analysis_metadata.get('shape_masks')),
              "verification": verification,
              "real_model_executed": True, "quality_adopted": False,
              "media_rendered_or_played":False,
              "physical_12gb_tested": False, "windows": windows}
    atomic_json(args.job / "technical-acceptance.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
