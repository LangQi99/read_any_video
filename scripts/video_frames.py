#!/usr/bin/env python3
import os
import sys

for _v in ("OPENCV_FFMPEG_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import subprocess
import tempfile


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _ensure_deps():
    required = {"cv2": "opencv-python-headless", "numpy": "numpy"}
    missing = [pkg for mod, pkg in required.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    log(f"[video_frames] Installing: {', '.join(missing)}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user",
                        "--break-system-packages", "--quiet"] + missing, check=True)
        importlib.invalidate_caches()
    except Exception as e:
        log(f"[video_frames] Install failed: {e}")
        sys.exit(3)


_ensure_deps()

import cv2
import numpy as np

cv2.setNumThreads(1)


def open_video(path):
    if not os.path.exists(path):
        log(f"ERROR: file not found: {path}")
        sys.exit(2)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log(f"ERROR: cannot open video: {path}")
        sys.exit(2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or fps > 240:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total / fps if total > 0 else 0.0
    return cap, fps, total, duration, w, h


def save_frame(cap, fps, t, outdir, quality, max_w):
    fidx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    if max_w and w > max_w:
        nh = int(h * max_w / w)
        frame = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)
    fn = os.path.join(outdir, f"t{t:07.2f}s.jpg")
    cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return fn


def motion_signature(frame, grid=16, blur=3):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        g = cv2.GaussianBlur(g, (blur * 2 + 1, blur * 2 + 1), 0)
    small = cv2.resize(g, (grid, grid), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def analyze_motion(cap, fps, t_start, t_end, analyze_fps=10.0, grid=16):
    step = max(1, int(round(fps / analyze_fps)))
    f0 = int(round(t_start * fps))
    f1 = int(round(t_end * fps))
    samples = []
    prev = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fidx = f0
    next_proc = f0
    last_log_t = -1e9
    span = max(1e-6, (f1 - f0) / fps)
    while fidx <= f1:
        if not cap.grab():
            break
        if fidx >= next_proc:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            sig = motion_signature(frame, grid=grid)
            t = fidx / fps
            score = 0.0 if prev is None else float(np.mean(np.abs(sig - prev)))
            samples.append((round(t, 3), round(score, 3)))
            prev = sig
            next_proc += step
            if t - last_log_t >= 5.0:
                log(f"  analyzing… {t - t_start:.0f}s / {span:.0f}s")
                last_log_t = t
        fidx += 1
    return samples


def auto_threshold(scores):
    nz = [s for s in scores if s > 0.01]
    if not nz:
        return 0.5
    med = float(np.median(nz))
    return max(0.8, med * 0.5)


def segment_motion(samples, thr):
    if not samples:
        return []
    segs = []
    cur_kind = None
    cur_t0 = samples[0][0]
    cur_peak = 0.0
    prev_t = samples[0][0]
    for t, s in samples:
        kind = "active" if s >= thr else "static"
        if cur_kind is None:
            cur_kind, cur_t0, cur_peak = kind, t, s
        elif kind != cur_kind:
            segs.append((cur_t0, prev_t, cur_kind, round(cur_peak, 2)))
            cur_kind, cur_t0, cur_peak = kind, t, s
        else:
            cur_peak = max(cur_peak, s)
        prev_t = t
    segs.append((cur_t0, prev_t, cur_kind, round(cur_peak, 2)))
    return segs


def merge_short(segs, min_static=0.4):
    if not segs:
        return segs
    out = [segs[0]]
    for seg in segs[1:]:
        t0, t1, kind, peak = seg
        if kind == "static" and (t1 - t0) < min_static and out:
            pt0, pt1, pkind, ppeak = out[-1]
            if pkind == "active":
                out[-1] = (pt0, t1, "active", ppeak)
                continue
        out.append(seg)
    merged = [out[0]]
    for seg in out[1:]:
        if seg[2] == merged[-1][2]:
            p = merged[-1]
            merged[-1] = (p[0], seg[1], p[2], max(p[3], seg[3]))
        else:
            merged.append(seg)
    return merged


def pick_scan_frames(segs, density):
    times = []
    for t0, t1, kind, peak in segs:
        if kind != "active":
            continue
        dur = t1 - t0
        if dur <= 0:
            times.append(round(t0, 3))
            continue
        n = max(3, int(round(dur * density)))
        for i in range(n):
            t = t0 + dur * i / (n - 1) if n > 1 else t0
            times.append(round(t, 3))
    return sorted(set(times))


def fmt_timeline(segs):
    lines = []
    for t0, t1, kind, peak in segs:
        tag = "active" if kind == "active" else "static"
        extra = f" (peak {peak})" if kind == "active" else ""
        lines.append(f"  {t0:6.2f}s - {t1:6.2f}s  [{tag}]{extra}")
    return "\n".join(lines)


def cmd_scan(args):
    cap, fps, total, duration, w, h = open_video(args.video)
    t_start = args.start if args.start is not None else 0.0
    t_end = args.end if args.end is not None else duration
    if t_end <= 0:
        t_end = duration if duration > 0 else 1e9

    log(f"Video: {args.video}")
    log(f"  {w}x{h}, {fps:.2f}fps, {duration:.2f}s, {total} frames")
    log(f"  scan range {t_start:.2f}s - {t_end:.2f}s")

    samples = analyze_motion(cap, fps, t_start, t_end,
                             analyze_fps=args.analyze_fps, grid=args.grid)
    scores = [s for _, s in samples]
    thr = args.threshold if args.threshold is not None else auto_threshold(scores)
    segs = merge_short(segment_motion(samples, thr))

    active = [s for s in segs if s[2] == "active"]
    static_total = sum(t1 - t0 for t0, t1, k, _ in segs if k == "static")

    outdir = args.outdir or tempfile.mkdtemp(prefix="vframes_scan_")
    os.makedirs(outdir, exist_ok=True)

    times = pick_scan_frames(segs, args.density)
    frames = []
    for t in times:
        fn = save_frame(cap, fps, t, outdir, args.quality, args.max_width)
        if fn:
            frames.append({"t": t, "path": fn})
    cap.release()

    result = {
        "mode": "scan",
        "video": os.path.abspath(args.video),
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "resolution": [w, h],
        "scan_range": [round(t_start, 3), round(t_end, 3)],
        "motion_threshold": round(thr, 3),
        "static_seconds_total": round(static_total, 2),
        "active_segments": [{"start": t0, "end": t1, "peak": peak}
                            for t0, t1, k, peak in active],
        "timeline": [{"start": t0, "end": t1, "kind": k, "peak": peak}
                     for t0, t1, k, peak in segs],
        "frames": frames,
        "outdir": outdir,
    }

    log("\n=== Motion Timeline ===")
    log(fmt_timeline(segs))
    log(f"\nStatic: {static_total:.2f}s folded (no frames extracted).")
    log(f"Active: {len(active)} segments, {len(frames)} frames -> {outdir}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_zoom(args):
    cap, fps, total, duration, w, h = open_video(args.video)
    if args.start is None or args.end is None:
        log("ERROR: zoom requires --start and --end")
        sys.exit(2)
    t_start = max(0.0, args.start)
    t_end = min(args.end, duration if duration > 0 else args.end)

    log(f"Video: {args.video}  ({fps:.2f}fps, {duration:.2f}s)")
    log(f"  zoom {t_start:.2f}s - {t_end:.2f}s @ {args.density}fps")

    outdir = args.outdir or tempfile.mkdtemp(prefix="vframes_zoom_")
    os.makedirs(outdir, exist_ok=True)

    dur = max(0.0, t_end - t_start)
    n = max(2, int(round(dur * args.density)) + 1)
    times = sorted(set(round(t_start + dur * i / (n - 1), 3) for i in range(n)))

    frames = []
    for t in times:
        fn = save_frame(cap, fps, t, outdir, args.quality, args.max_width)
        if fn:
            frames.append({"t": t, "path": fn})
    cap.release()

    result = {
        "mode": "zoom",
        "video": os.path.abspath(args.video),
        "fps": round(fps, 3),
        "zoom_range": [round(t_start, 3), round(t_end, 3)],
        "density_fps": args.density,
        "frames": frames,
        "outdir": outdir,
    }
    log(f"{len(frames)} frames -> {outdir}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_grid(args):
    cap, fps, total, duration, w, h = open_video(args.video)
    t_start = args.start if args.start is not None else 0.0
    t_end = args.end if args.end is not None else duration
    if t_end <= 0:
        t_end = duration if duration > 0 else 1e9
    rows, cols = max(1, args.rows), max(1, args.cols)
    n = rows * cols
    span = max(0.0, t_end - t_start)
    if n > 1 and span > 0:
        times = [round(t_start + span * i / (n - 1), 3) for i in range(n)]
    else:
        times = [round(t_start, 3)] * n

    log(f"Video: {args.video}  ({fps:.2f}fps, {duration:.2f}s)")
    log(f"  grid {rows}x{cols}, range {t_start:.2f}s-{t_end:.2f}s, cell {args.cell_width}px")

    cell_w = max(80, args.cell_width)
    imgs, cells = [], []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((max(1, int(cell_w * h / max(1, w))), cell_w, 3), np.uint8)
        fh, fw = frame.shape[:2]
        cell = cv2.resize(frame, (cell_w, max(1, int(fh * cell_w / fw))),
                          interpolation=cv2.INTER_AREA)
        label = f"{t:.1f}s"
        cv2.putText(cell, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        imgs.append(cell)
        cells.append({"idx": i, "t": t, "row": i // cols, "col": i % cols})
    cap.release()

    cell_h = min(im.shape[0] for im in imgs)
    imgs = [im[:cell_h] for im in imgs]
    grid_img = np.vstack([np.hstack(imgs[r * cols:(r + 1) * cols]) for r in range(rows)])

    outdir = args.outdir or tempfile.mkdtemp(prefix="vframes_grid_")
    os.makedirs(outdir, exist_ok=True)
    grid_path = os.path.join(outdir, "grid.jpg")
    cv2.imwrite(grid_path, grid_img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])

    result = {
        "mode": "grid",
        "video": os.path.abspath(args.video),
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "range": [round(t_start, 3), round(t_end, 3)],
        "rows": rows, "cols": cols,
        "grid_path": grid_path,
        "cells": cells,
        "outdir": outdir,
    }
    log(f"\nGrid -> {grid_path}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("video")
    common.add_argument("--start", type=float, default=None)
    common.add_argument("--end", type=float, default=None)
    common.add_argument("--outdir", default=None)
    common.add_argument("--quality", type=int, default=70)
    common.add_argument("--max-width", type=int, default=900)

    ps = sub.add_parser("scan", parents=[common])
    ps.add_argument("--analyze-fps", type=float, default=10.0)
    ps.add_argument("--grid", type=int, default=16)
    ps.add_argument("--threshold", type=float, default=None)
    ps.add_argument("--density", type=float, default=2.0)
    ps.set_defaults(func=cmd_scan)

    pz = sub.add_parser("zoom", parents=[common])
    pz.add_argument("--density", type=float, default=8.0)
    pz.set_defaults(func=cmd_zoom)

    pg = sub.add_parser("grid", parents=[common])
    pg.add_argument("--rows", type=int, default=3)
    pg.add_argument("--cols", type=int, default=3)
    pg.add_argument("--cell-width", type=int, default=320)
    pg.set_defaults(func=cmd_grid)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
