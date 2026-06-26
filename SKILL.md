---
name: read_any_video
description: >-
  LLM 读不了视频 —— 本 skill 帧间差分分析
  把视频拆成模型能读的关键帧 + 带时间戳的运动时间线，
  让模型得以描述、总结、定位视频中的任意瞬间。
  触发场景：提供视频文件 / URL（.mp4/.mov/.avi/gif 等）
  Give LLMs eyes for video: extract key frames + a timestamped motion timeline
  so the model can describe, summarize, or locate moments in a video.
---

# Read Any Video · 让 LLM "看见"视频

> **LLM 读不了视频。** 这个 skill 把视频拆成它能读的东西：关键帧图片 + 运动时间线。

模型无法直接播放或理解 `.mp4`。本工具用**帧间差分分析**找出
哪里静止、哪里有动作、关键瞬间在第几秒，只抽出真正有信息量的帧
交给模型去看——于是模型就能描述、总结、定位视频里的任意瞬间。
除了内置脚本，**零额外依赖**。

## ⚡ 工作流：先扫描，再聚焦

```
1. 拿到视频   → 下载 URL / 用本地路径 / 解压或重命名伪装文件
2. scan 扫描  → 运动时间线 + 活跃片段的稀疏关键帧
3. zoom 聚焦  → 对感兴趣的区间做高密度抽帧（可选）
4. 总结       → 输出带时间戳的客观叙事
```

一句话：**别一上来就暴力抽几百帧**，先 scan 看全局，再 zoom 抠细节。

## 🛠 命令一览

### scan · 默认入口

```bash
python3 <SKILL_DIR>/scripts/video_frames.py scan <video>
```

逐帧计算运动分数，自动切分为「静止 / 活跃」片段，
**只在活跃片段抽帧**——把算力花在真正有内容的地方。

- **stderr**：运动时间线（请先读这里）
- **stdout**：JSON，含 `timeline`、`active_segments`、`frames[]{path, t}`

可选参数：`--start/--end`、`--density N`（帧/秒，默认 2）、
`--threshold`、`--max-width`（默认 900）。

### zoom · 聚焦细节

```bash
python3 <SKILL_DIR>/scripts/video_frames.py zoom <video> --start 10 --end 12 --density 8
```

在一个窄区间内做高密度抽帧。scan 锁定可疑区域后再用，
还能在更小的区间上反复聚焦，层层逼近关键瞬间。

### grid · 全局缩略图

```bash
python3 <SKILL_DIR>/scripts/video_frames.py grid <video>   # --rows 4 --cols 4
```

均匀采样并拼成一张带时间戳的九宫格大图。
长视频或第一眼总览时特别好用。

只有当用户提供了背景预期时，才去做「与预期对比」的判断。

## 📝 注意事项

- 脚本路径：`<SKILL_DIR>/scripts/video_frames.py`（绝对路径）
- 依赖：`opencv-python-headless` + `numpy`（自动安装）
- 线程数固定为 1 以兼容沙箱环境——**请勿删除**
- 帧输出到 `/tmp`，读完即可丢弃
- 伪装文件（.txt/.bin/.zip）：先解压或重命名再用，OpenCV 按内容而非扩展名识别
- 永远先 scan / grid，**切勿一次性暴力抽取几百帧**
