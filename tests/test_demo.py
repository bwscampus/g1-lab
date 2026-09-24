import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from agent import Agent
from decider import HFDecider
from demo import (HISTORICAL, HISTORICAL_ACTION, HISTORICAL_VIDEO, ImagePart, ModelSelector, Request, TextPart,
                  UniformSelector, VideoPart, build_request, bundle_parts, compile_run, compile_video, content_records,
                  load_bundle, load_manifest, prepare, save_input, thin, validate_selection, write_bundle, _main)
from episode import EpisodeWriter
from hf import HFClient
from run import build_parser, main, run
from skills import SKILLS, menu
from tests.test_agent import Sequence, check


def png(path, color=(10, 20, 30)):
    import cv2
    cv2.imwrite(str(path), np.full((8, 8, 3), color[::-1], np.uint8))
    return path


def test_manifest_rules(tmp_path):
    img = png(tmp_path / "a.png")

    def load(data):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(data))
        return load_manifest(p)

    m = load({"instruction": "find it", "content": ["look", {"image": "a.png", "label": "goal"}]})
    assert m.instruction == "find it" and m.content == (TextPart("look"), ImagePart(img.resolve(), "goal", None))
    m = load({"content": ["find it"]})                                   # instruction from the text
    assert m.instruction == "find it"
    (tmp_path / "v.mp4").write_bytes(b"x")
    m = load({"instruction": "g", "content": [{"video": "v.mp4", "mode": "video+action", "label": "d"}]})
    assert m.content[0] == VideoPart((tmp_path / "v.mp4").resolve(), "d", None, "video+action")
    for bad in [[], {"content": []}, {"content": [""]}, {"content": [3]}, {"content": [{"image": "a.png", "video": "v.mp4"}]},
                {"content": [{"image": "missing.png"}]}, {"content": [{"image": "a.png", "label": " "}]},
                {"content": [{"image": "a.png", "detail": "huge"}]}, {"content": [{"video": "v.mp4", "mode": "actions"}]},
                {"content": [{"image": "a.png", "mode": "video"}]}, {"content": [{"image": "a.png"}]}]:
        with pytest.raises(ValueError):
            load(bad)
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "bad.json")
    assert content_records(m.content) == [{"video": str((tmp_path / "v.mp4").resolve()), "mode": "video+action", "label": "d"}]


def test_build_request(tmp_path):
    with pytest.raises(ValueError):
        build_request(None)
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"x")
    r = build_request("find the mug", demo=video)
    assert r.content == (VideoPart(video.resolve(), None, None, "video"),) and r.record()["content"][0]["mode"] == "video"
    with pytest.raises(ValueError):
        build_request("g", demo=video, mode="video+action")            # a video alone has no action data
    with pytest.raises(ValueError):
        build_request("g", demo=tmp_path / "missing.mp4")
    ref = png(tmp_path / "mug.png")
    r = build_request("g", refs=[ref])
    assert r.content == (ImagePart(ref.resolve(), "mug"),)
    with pytest.raises(ValueError):
        build_request("g", refs=[video])
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"instruction": "from manifest", "content": ["from manifest"]}))
    r = build_request(None, manifest=m)
    assert r.instruction == "from manifest" and r.manifest is not None and r.record()["manifest"]["source"] == str(m.resolve())
    assert build_request("cli wins", manifest=m).instruction == "cli wins"
    assert thin(5, 12) == [0, 1, 2, 3, 4] and thin(100, 3) == [0, 50, 99] and thin(3, 1) == [0]


def recorded_run(tmp_path):
    pytest.importorskip("cv2")
    dec = Sequence([("turn", {"angle_deg": 30}), ("tpose", {"hold": 1.0, "rise": 1.0}),
                    ("done", {"summary": "s", "hindsight": "h"})])
    rec = EpisodeWriter(tmp_path / "runs", env="sim", goal="find the red ball", model=dec.model,
                        skills=list(SKILLS), threaded=False)
    agent = Agent("find the red ball", dec, menu(True), recorder=rec, verdict=lambda: "success")
    assert run(agent, check(), max_time=120) is True
    agent.close()
    return rec.dir


def test_recorded_run_becomes_a_bundle(tmp_path):
    run_dir = recorded_run(tmp_path)
    b = compile_run(run_dir, "video", max_frames=3)
    assert b["mode"] == "video" and b["outcome"] == "success" and b["demonstrator"] == "policy_rollout"
    assert len(b["keyframes"]) == 3 and [k["label"]["index"] for k in b["keyframes"]] == [0, 1, 2]
    assert b["keyframes"][0]["label"]["stage"] == "turn completed" and b["keyframes"][-1]["label"]["stage"] == "done done"
    assert all(Path(k["image"]).is_file() and "action" not in k for k in b["keyframes"])
    a = compile_run(run_dir, "video+action")
    assert len(a["keyframes"]) == 3 and a["keyframes"][0]["action"]["skill"] == "turn"
    act = a["keyframes"][0]["action"]
    assert act["arguments"]["angle_deg"] == 30.0 and act["outcome"] == "completed" and act["base_pose_cmd"] == [0.0, 0.0, 0.0]
    rows = act["joint_pos_samples"]
    assert rows[0]["q"][0] != "=" and len(rows) >= 2 and "=" in rows[1]["q"]      # 1 Hz, unchanged values as "="
    assert a["keyframes"][1]["action"]["skill"] == "tpose" and a["keyframes"][2]["action"]["skill"] == "done"
    out = write_bundle(a, tmp_path / "bundle")
    loaded = load_bundle(out)
    assert loaded["keyframes"][0]["image"] == str((tmp_path / "bundle" / "kf-000.png").resolve())
    assert json.loads(out.read_text())["keyframes"][0]["image"] == "kf-000.png"      # portable: relative paths
    parts = bundle_parts(loaded, "video+action")
    assert parts[0] == TextPart(HISTORICAL + HISTORICAL_ACTION)
    assert parts[1].text.startswith("Demonstration: ") and "policy_rollout" in parts[1].text
    assert sum(isinstance(p, ImagePart) for p in parts) == 3 and parts[-1].text.endswith("END OF DEMONSTRATION.")
    assert '"action":{"skill":"turn"' in parts[2].text and parts[3].label.startswith("keyframe 0, t=")
    video_parts = bundle_parts(loaded, "video")
    assert video_parts[0].text.endswith(HISTORICAL_VIDEO) and "action" not in video_parts[2].text


def test_prepare_and_turn_zero(tmp_path):
    run_dir = recorded_run(tmp_path)
    request = build_request("find the red ball", demo=run_dir)
    assert request.content[0].mode == "video+action"                     # the default for a recorded run
    prepared, reports = prepare(request, tmp_path / "input", max_frames=2)
    assert reports[0]["keyframes"] == 2 and reports[0]["mode"] == "video+action"
    assert (tmp_path / "input" / "video-000" / "demo.json").is_file()
    images = [p for p in prepared.content if isinstance(p, ImagePart)]
    assert len(images) == 2 and all(str(p.path).startswith(str(tmp_path / "input")) for p in images)
    assert prepare(Request("g", ()), tmp_path / "x") == (Request("g", ()), [])
    # the same bundle reloads without recompiling, and a video-mode request drops the actions
    again, _ = prepare(build_request("g", demo=tmp_path / "input" / "video-000" / "demo.json", mode="video"), tmp_path / "input2")
    assert again.content[0].text.endswith(HISTORICAL_VIDEO) and "action" not in again.content[2].text
    # saved and reloaded: the archived request is the same content
    saved = save_input(prepared, tmp_path / "input")
    m = load_manifest(saved)
    assert m.instruction == "find the red ball" and content_records(m.content) == content_records(prepared.content)
    assert (tmp_path / "input" / "images").exists() is False            # images already lived under input/
    # turn 0 carries the demonstration, turn 1 does not; the session keeps the demo images
    calls = []
    turn_json = '{"name": "turn", "arguments": {"angle_deg": 45, "note": "n"}}'
    done_json = '{"name": "done", "arguments": {"summary": "s", "hindsight": "h"}}'

    def transport(url, headers, body, timeout):
        calls.append(body)
        return iter([json.dumps({"choices": [{"delta": {"content": turn_json if len(calls) == 1 else done_json}}]}), "[DONE]"])

    dec = HFDecider(HFClient("m/vl", "hf_x", transport=transport), live_image_window=1)
    dec.threaded = False
    agent = Agent("find the red ball", dec, menu(True), content=prepared.content, max_decisions=5)
    assert run(agent, check(), max_time=120) is True and agent.result == "completed"
    user0 = calls[0]["messages"][1]["content"]
    assert user0[0]["text"] == HISTORICAL + HISTORICAL_ACTION
    assert sum(b["type"] == "image_url" for b in user0) == 3               # 2 demo images + the camera
    assert user0[-2]["text"] == "Camera image: head" and json.loads(user0[-3]["text"])["instruction"] == "find the red ball"
    assert user0[-4]["text"].endswith("END OF DEMONSTRATION.") and user0[-5]["type"] == "image_url"
    user1 = calls[1]["messages"][3]["content"]
    assert user1[0]["type"] == "text" and json.loads(user1[0]["text"])["extra"]["env_step"] == 1
    assert sum(b["type"] == "image_url" for b in user1) == 1
    pruned0 = calls[1]["messages"][1]["content"]                          # window 1: only the camera image went
    assert sum(b["type"] == "image_url" for b in pruned0) == 2 and "omitted" in pruned0[-1]["text"]


def synthetic_video(path, seconds=3.0, fps=10):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        w.write(np.full((48, 64, 3), (min(255, i * 3), 0, 0), np.uint8))
    w.release()
    return path


def test_video_uniform_selection_and_cache(tmp_path):
    pytest.importorskip("cv2")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg not installed")
    video = synthetic_video(tmp_path / "walk.mp4")
    b = compile_video(video, "find the mug", tmp_path / "kf", selector=UniformSelector(4), cache_dir=tmp_path / "cache")
    assert b["mode"] == "video" and len(b["keyframes"]) == 4 and b["metadata"]["cache_hit"] is False
    assert b["keyframes"][0]["t_s"] == 0.0 and b["keyframes"][-1]["t_s"] > 2.5 and b["metadata"]["candidates"] == 7
    assert all(Path(k["image"]).is_file() for k in b["keyframes"]) and b["metadata"]["width"] == 64
    b2 = compile_video(video, "find the mug", tmp_path / "kf2", selector=UniformSelector(4), cache_dir=tmp_path / "cache")
    assert b2["metadata"]["cache_hit"] is True and b2["metadata"]["cache_key"] == b["metadata"]["cache_key"]
    b3 = compile_video(video, "another goal", tmp_path / "kf3", selector=UniformSelector(4), cache_dir=tmp_path / "cache")
    assert b3["metadata"]["cache_hit"] is False                            # the instruction is part of the key
    request = build_request("find the mug", demo=video)
    prepared, reports = prepare(request, tmp_path / "input", selector=UniformSelector(3), cache_dir=tmp_path / "cache")
    assert reports[0]["keyframes"] == 3 and prepared.content[0].text.endswith(HISTORICAL_VIDEO)
    with pytest.raises(ValueError):
        prepare(Request("g", (VideoPart(video, None, None, "video+action"),)), tmp_path / "y")


def test_video_model_selection(tmp_path):
    pytest.importorskip("cv2")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    video = synthetic_video(tmp_path / "walk.mp4", seconds=2.0)
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body)
        reply = {"selected": [{"index": 0, "reason": "start", "stage": "start", "result": ""},
                              {"index": 4, "reason": "end", "stage": "arrived", "result": "goal in view"}],
                 "summary": "walked straight to the goal"}
        return iter([json.dumps({"choices": [{"delta": {"content": json.dumps(reply)}}]}),
                     json.dumps({"choices": [], "usage": {"prompt_tokens": 9}}), "[DONE]"])

    sel = ModelSelector(HFClient("m/vl", "hf_x", transport=transport), max_frames=6)
    b = compile_video(video, "find the mug", tmp_path / "kf", selector=sel, cache_dir=tmp_path / "cache")
    assert len(calls) == 1 and calls[0]["response_format"]["type"] == "json_schema"
    user = calls[0]["messages"][1]["content"]
    assert "User's final task: find the mug" in user[0]["text"] and sum(b_["type"] == "image_url" for b_ in user) == 5
    assert [k["label"]["stage"] for k in b["keyframes"]] == ["start", "arrived"] and b["summary"] == "walked straight to the goal"
    assert sel.calls[0]["usage"] == {"prompt_tokens": 9} and sel.calls[0]["phase"] == "demo"
    assert b["metadata"]["selector"]["model"] == "m/vl"
    for bad in [{"selected": [], "summary": "s"}, {"selected": [{"index": 9, "reason": "r", "stage": "", "result": ""}], "summary": "s"},
                {"selected": [{"index": 1, "reason": "r", "stage": "", "result": ""}] * 2, "summary": "s"},
                {"selected": [{"index": 1, "reason": "", "stage": "", "result": ""}], "summary": "s"}]:
        with pytest.raises(ValueError):
            validate_selection(bad, 5, 6)


def test_cli_prepare_show_and_run_flags(tmp_path, capsys, monkeypatch):
    run_dir = recorded_run(tmp_path)
    assert _main(["prepare", str(tmp_path / "out"), "--goal", "find the red ball", "--demo", str(run_dir),
                  "--demo-frames", "2", "--demo-select", "uniform"]) == 0
    out = capsys.readouterr().out
    assert "2 keyframe(s) (video+action)" in out and (tmp_path / "out" / "demo.json").is_file()
    assert _main(["show", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "HISTORICAL DEMONSTRATION" in out and out.count("[image]") == 2
    with pytest.raises(SystemExit):
        _main(["prepare", str(tmp_path / "o2"), "--goal", "g"])                    # nothing to compile
    args = build_parser().parse_args(["--policy", "search", "--demo", "x.mp4", "--ref", "a.png", "--ref", "b.png",
                                      "--demo-mode", "video", "--demo-frames", "5", "--input-json", "m.json"])
    assert args.demo == "x.mp4" and args.ref == ["a.png", "b.png"] and args.demo_frames == 5
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("G1_VISION_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["--env", "sim", "--headless", "--policy", "search", "--goal", "g", "--demo", str(tmp_path / "none.mp4"), "--no-log"])
    assert "demonstration not found" in capsys.readouterr().err
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"instruction": "from the manifest"}))
    with pytest.raises(SystemExit):
        main(["--env", "sim", "--headless", "--policy", "search", "--input-json", str(m), "--no-log"])
    assert "HF_TOKEN" in capsys.readouterr().err                                    # the goal came from the manifest
