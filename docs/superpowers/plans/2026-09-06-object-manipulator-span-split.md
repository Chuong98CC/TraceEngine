# Object/Manipulator Sampling & Trace Span Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample and trace **object** init points only over a subtask's transport span (gripper close → gripper open, the 2nd..2nd-to-last keyframe) while the **manipulator** keeps today's full-span behavior (start → last frame).

**Architecture:** Step 3a records each prompt's role (`object`/`manipulator` — the `meta/subtasks.csv` column it was read from) as `prompt_roles` in the detections JSON. Step 3b reads the role per prompt and matches object prompts on `keys[1:-1]` only (everything else keeps the full list). Step 4d derives each pass's trace window from its prompts' npz `frame_indices` — the last Step-2 stem ≤ the earliest first keyframe to the first stem ≥ the latest last keyframe (a pure helper `span_stems` in `keyframe_utils`) — so object traces run stem-before-close → stem-after-open, and full-span prompts recover exactly today's behavior. No Step-1 or Step-2 changes; the `run_step3_init_points.py` driver needs no functional change (it already forwards `--data-root` and the detections JSON).

**Tech Stack:** Python ≥3.12.1 <3.13, numpy, torch.export runtimes; Step 3a runs under `.venv-rexomni/bin/python` (Python 3.10 env), Step 3b/4d and tests under the main env.

**Spec:** No spec file — design was agreed in conversation (2026-09-06 brainstorming). Requirements: (1) object prompts match `keys[1:-1]` of their subtask — "always", even when that leaves < 2 frames → empty output `insufficient keyframes`; (2) manipulator and any prompt without a recorded role keep the full list (folder-mode and older-JSON runs are unchanged); (3) step-4 object traces start at the stem just before the close keyframe (object is static until close, so close-frame pixels are exact there) and end at the stem just after the open keyframe, on the existing stride-4 grid — **no Step-2 change**; (4) manipulator traces unchanged (subtask first stem → last stem).

## Global Constraints

- Repo root: `/home/chuong/workspace/depth_models/DepthModels`. All paths below are relative to it.
- Roles on disk are the literal csv column names `"object"` / `"manipulator"` (see `SUBTASK_PROMPT_COLUMNS` in `src/utils/keyframe_utils.py:42`).
- Step 3a runs with `.venv-rexomni/bin/python`; Step 3b / Step 4d / tests run with the main env (`python` / `uv run --extra dev pytest`).
- Test suite: CPU-only; run `uv run --extra dev pytest tests/ -q`. `tests/` currently holds only `tests/test_image_io.py` — new test files follow its plain-pytest style.
- **The working tree has unrelated dirty files — never `git add -A`:** `scripts/astribot/extract_frames.sh`, `scripts/general_test/module/infer_moge3.sh`, `tools/general_test/module/infer_moge3.py` carry the user's in-progress work. Stage only the exact paths each commit names.
- Every commit message ends with the `Co-Authored-By: Claude <noreply@anthropic.com>` trailer line.
- The real-data verification (Task 6) uses `/data/astribot_making_coffee_vlva_full` (`--data-root`) and repo id `Kronze157/astribot_making_coffee_vlva_full`; key-frames live under `<data-root>/eps_data/key_frames`, Step-2 geometry under `<data-root>/eps_data/depth_pose`, detections under `<data-root>/eps_data/detections`, init points under `<data-root>/eps_data/init_points`.
- Old-format inputs stay supported: detections JSONs without `prompt_roles` (folder mode, pre-change runs) make Step 3b treat every prompt as full-span — identical to today's output.

## File Structure

- Modify `src/utils/keyframe_utils.py` — two small pure helpers: `subtask_prompt_roles` (row → aligned prompts+roles) and `span_stems` (stem grid clipped to a keyframe envelope), siblings of the existing `subtask_prompts` / `cap_keyframes`.
- Modify `tools/general_test/pipeline/run_object_detection.py` (Step 3a) — record `prompt_roles` next to `prompts` per subtask (episode mode only).
- Modify `tools/general_test/pipeline/run_object_init_points.py` (Step 3b) — thread the role into `_process_prompt`; object prompts match on `keyframes[1:-1]`; meta gains `"role"` and the `"segment"` envelope comes from the full keyframe list.
- Modify `tools/astribot/run_step4_traces.py` (Step 4d) — per-pass trace window via `span_stems`; anchor candidates inside the window; `_usable_at` falls back to the first keyframe column for the window's leading (pre-close) stem.
- Create `tests/test_span_stems.py`; extend `tests/test_subtask_meta.py` (append `subtask_prompt_roles` tests — the file already pins `load_subtask_meta` / `subtask_prompts`).
- Modify `README.md`, `docs/astribot/astribot_traces.md`, `docs/general_test/general_test.md`, `docs/general_test/pipeline/step3.md`, `docs/general_test/module/romav2.md`, `docs/general_test/module/rexomni.md` — span wording.
- No change: `tools/astribot/run_step3_init_points.py`, `tools/astribot/extract_frames.py`, `tools/astribot/run_step2_depth_stream.py`, `tools/general_test/pipeline/run_e2e_init_points.py`.

---

### Task 1: `subtask_prompt_roles` helper in `keyframe_utils`

**Files:**
- Modify: `src/utils/keyframe_utils.py` (around the existing `subtask_prompts`, lines 261–266)
- Modify: `tests/test_subtask_meta.py` (append tests; the file already pins `load_subtask_meta` / `subtask_prompts` behaviour)

**Interfaces:**
- Consumes: `SUBTASK_PROMPT_COLUMNS = ("object", "manipulator")` (already at `keyframe_utils.py:42`).
- Produces: `subtask_prompt_roles(row: dict[str, str] | None) -> tuple[list[str], list[str]]` — prompts in column order `[object, manipulator]`, roles aligned (the column each prompt came from), empty cells dropped; `subtask_prompts(row)` re-implemented as `subtask_prompt_roles(row)[0]` (unchanged semantics — other callers keep working).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_subtask_meta.py` (the module docstring at the top already names `subtask_prompts`; extend its mention to `subtask_prompt_roles` and add the import):

```python
from utils.keyframe_utils import (
    load_subtask_meta,
    subtask_prompt_roles,
    subtask_prompts,
)
```

and append at the end of the file:

```python
def test_subtask_prompt_roles_aligned_object_then_manipulator():
    row = {"object": "brown cup",
           "manipulator": "left robot arm's black grippers"}
    prompts, roles = subtask_prompt_roles(row)
    assert prompts == ["brown cup", "left robot arm's black grippers"]
    assert roles == ["object", "manipulator"]
    # subtask_prompts keeps its historical return value
    assert subtask_prompts(row) == prompts


def test_subtask_prompt_roles_partial_row():
    row = {"manipulator": "left robot arm's black grippers"}
    prompts, roles = subtask_prompt_roles(row)
    assert prompts == ["left robot arm's black grippers"]
    assert roles == ["manipulator"]


def test_subtask_prompt_roles_empty_and_missing_row():
    assert subtask_prompt_roles({}) == ([], [])
    assert subtask_prompt_roles(None) == ([], [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_subtask_meta.py -q`
Expected: FAIL — `ImportError: cannot import name 'subtask_prompt_roles'` (the pre-existing `subtask_prompts` tests still pass).

- [ ] **Step 3: Implement the helper**

In `src/utils/keyframe_utils.py`, replace the body of `subtask_prompts` (lines 261–266) with:

```python
def subtask_prompt_roles(row: dict[str, str] | None
                         ) -> tuple[list[str], list[str]]:
    """([object, manipulator] text prompts of one annotation row, the role
    of each prompt — the meta/subtasks.csv column it was read from).
    Empty cells are dropped; prompts and roles stay aligned."""
    row = row or {}
    return ([row[c] for c in SUBTASK_PROMPT_COLUMNS if row.get(c)],
            [c for c in SUBTASK_PROMPT_COLUMNS if row.get(c)])


def subtask_prompts(row: dict[str, str] | None) -> list[str]:
    """RexOmni/SAM3 text prompts of one annotation row: the non-empty
    [object, manipulator] values ([] when the row is missing or carries
    neither). See ``subtask_prompt_roles``."""
    return subtask_prompt_roles(row)[0]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/test_subtask_meta.py -q`
Expected: PASS (all tests of the file, including the three new ones).

- [ ] **Step 5: Commit**

```bash
git add src/utils/keyframe_utils.py tests/test_subtask_meta.py
git commit -m "feat(keyframe_utils): add subtask_prompt_roles(row) aligned prompts+roles

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Step 3a records `prompt_roles` in the detections JSON

**Files:**
- Modify: `tools/general_test/pipeline/run_object_detection.py` (import at lines 69–82, module docstring lines 23–25, episode record lines 436–449)

**Interfaces:**
- Consumes: `subtask_prompt_roles` from Task 1.
- Produces: episode-mode detections JSON segment records gain an aligned `"prompt_roles": ["object" | "manipulator", ...]` array next to `"prompts"`. Folder mode (`--keyframes-dir`) intentionally writes **no** `prompt_roles` (free-form `--text-prompts`, roles unknowable) — Step 3b falls back to full-span there.

- [ ] **Step 1: Replace the import and update the docstring**

In `tools/general_test/pipeline/run_object_detection.py`:

1. In the `from utils.keyframe_utils import (...)` block (lines 69–82) replace `subtask_prompts,` with `subtask_prompt_roles,` (keep the list sorted).
2. In the module docstring, after the sentence "Object prompts are per sub-task: the [object, manipulator] of the sub-task's row in the dataset's meta/subtasks.csv, recorded in the JSON next to the detections." (lines 23–25), append:

```
Each prompt is recorded with its role — the column it was read from —
under ``prompt_roles`` (aligned with ``prompts``): Step 3b uses it to
sample object keypoints only between the sub-task's gripper close/open
key-frames. Folder mode takes --text-prompts instead and records no
roles. RexOmni needs its own ...
```

(Keep the rest of the paragraph — the RexOmni environment note — as-is.)

- [ ] **Step 2: Record the roles in episode mode**

In `_process_episode`, replace (lines 436–449):

```python
            prompts = subtask_prompts(self.meta.get(label))
            if not prompts:
                print(f"  [subtask {k:02d}] skip: no object/manipulator "
                      f"row for label {label} in meta/subtasks.csv")
                continue
            print(f"  [subtask {k:02d}] label {label}: {len(keys)} "
                  f"key-frames {keys}, prompts {prompts}")
            subtasks[str(k)] = {
                "subtask_index": label,
                "segment": [min(keys), max(keys) + 1],
                "keyframes": keys,
                "prompts": prompts,
                "detections": self._detect_segment(cam, k, keys, prompts),
            }
```

with:

```python
            row = self.meta.get(label)
            prompts, prompt_roles = subtask_prompt_roles(row)
            if not prompts:
                print(f"  [subtask {k:02d}] skip: no object/manipulator "
                      f"row for label {label} in meta/subtasks.csv")
                continue
            print(f"  [subtask {k:02d}] label {label}: {len(keys)} "
                  f"key-frames {keys}, prompts "
                  f"{dict(zip(prompts, prompt_roles))}")
            subtasks[str(k)] = {
                "subtask_index": label,
                "segment": [min(keys), max(keys) + 1],
                "keyframes": keys,
                "prompts": prompts,
                "prompt_roles": prompt_roles,
                "detections": self._detect_segment(cam, k, keys, prompts),
            }
```

- [ ] **Step 3: Note the folder-mode contract**

In `_process_folder` (line ~389, where `prompts = list(self.args.text_prompts)`), the JSON entry stays without a `prompt_roles` key. Add a one-line comment above `"prompts": prompts,`:

```python
                # folder mode has no meta/subtasks.csv row: prompts carry
                # no role, Step 3b samples them over all key-frames
```

- [ ] **Step 4: Syntax smoke check**

Run: `python -m py_compile tools/general_test/pipeline/run_object_detection.py`
Expected: exit 0, no output.

- [ ] **Step 5: Commit**

```bash
git add tools/general_test/pipeline/run_object_detection.py
git commit -m "feat(step3a): record aligned prompt_roles in the detections JSON

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Step 3b samples object prompts on the close..open keyframes only

**Files:**
- Modify: `tools/general_test/pipeline/run_object_init_points.py` (module docstring lines 22–25; `_process_episode` items loop lines 408–427; `_process_folder` items line 467–469; `_process_segment` 471–479; `_process_prompt` 481–497, meta dict 581–600)

**Interfaces:**
- Consumes: detections JSON `prompts` + `prompt_roles` (Task 2); old JSONs / folder mode have no roles → all prompts full-span.
- Produces: object-role prompts whose output npz `frame_indices` are the subtask's 2nd..2nd-to-last keyframe (`[close, open]` for the canonical 4-frame subtask — N=2), and `init_points.json` gains `"role"`. Manipulator/unknown prompts are byte-identical to today.

- [ ] **Step 1: Update the module docstring**

In the Step-3b module docstring (lines 22–25), replace:

```
Prompts are read per sub-task from the Step-3a detections JSON (Step 3a
recorded them from the dataset's meta/subtasks.csv in episode mode, or from
its --text-prompts in folder mode). There is no JSON-less fallback — run
Step 3a first. All checkpoints are the repo defaults.
```

with:

```
Prompts are read per sub-task from the Step-3a detections JSON (Step 3a
recorded them from the dataset's meta/subtasks.csv in episode mode —
together with each prompt's role, the column it was read from — or from
its --text-prompts in folder mode, without roles). An **object** prompt
is matched only between the sub-task's 2nd and 2nd-to-last key-frame
(the gripper close .. open transport span, where the object is static on
the dropped boundary frames anyway); the **manipulator** — and any prompt
whose role is unrecorded (folder mode, older Step-3a JSONs) — is matched
across all key-frames. There is no JSON-less fallback — run Step 3a first.
All checkpoints are the repo defaults.
```

- [ ] **Step 2: Thread roles through the episode items**

In `_process_episode`, replace (lines 418–427):

```python
            prompts = sub.get("prompts") or []
            if not prompts:
                print(f"  [subtask {k:02d}] skip: no prompts recorded "
                      f"in the Step-3a JSON (re-run Step 3a — prompts "
                      f"are per-sub-task now)")
                continue
            items.append((int(k), keys,
                          sub.get("detections") or {}, prompts))
        for k, keys, seg_dets, prompts in items:
            self._process_segment(k, keys, seg_dets, prompts)
```

with:

```python
            prompts = sub.get("prompts") or []
            if not prompts:
                print(f"  [subtask {k:02d}] skip: no prompts recorded "
                      f"in the Step-3a JSON (re-run Step 3a — prompts "
                      f"are per-sub-task now)")
                continue
            # roles aligned with prompts (the meta/subtasks.csv column each
            # prompt was read from); a missing/mismatched list means every
            # prompt samples over all key-frames, as before
            prompt_roles = sub.get("prompt_roles") or []
            if prompt_roles and len(prompt_roles) != len(prompts):
                print(f"  [subtask {k:02d}] warning: prompt_roles length "
                      f"mismatch — sampling every prompt over all "
                      f"key-frames")
                prompt_roles = []
            items.append((int(k), keys, sub.get("detections") or {},
                          prompts, prompt_roles))
        for k, keys, seg_dets, prompts, prompt_roles in items:
            self._process_segment(k, keys, seg_dets, prompts, prompt_roles)
```

- [ ] **Step 3: Thread roles through the folder item**

In `_process_folder`, replace (lines 467–469):

```python
        items = [(0, keys, sub.get("detections") or {}, prompts)]
        for k, keys, seg_dets, prompts in items:
            self._process_segment(k, keys, seg_dets, prompts)
```

with:

```python
        items = [(0, keys, sub.get("detections") or {}, prompts, [])]
        for k, keys, seg_dets, prompts, prompt_roles in items:
            self._process_segment(k, keys, seg_dets, prompts, prompt_roles)
```

- [ ] **Step 4: Update `_process_segment`**

Replace (lines 471–479):

```python
    def _process_segment(self, k: int, keys: list[int],
                         seg_dets: dict | None, prompts: list[str]) -> None:
        seg_dir = os.path.join(self.init_dir, f"ep{self.ep_idx:06d}",
                               f"subtask_{k:02d}")
        print(f"  [subtask {k:02d}] {len(keys)} key-frames {keys}, "
              f"prompts {prompts}")
        frames = [self._load_keyframe(k, t) for t in keys]
        for prompt in prompts:
            self._process_prompt(seg_dir, k, keys, frames, seg_dets, prompt)
```

with:

```python
    def _process_segment(self, k: int, keys: list[int],
                         seg_dets: dict | None, prompts: list[str],
                         prompt_roles: list[str] | None = None) -> None:
        seg_dir = os.path.join(self.init_dir, f"ep{self.ep_idx:06d}",
                               f"subtask_{k:02d}")
        print(f"  [subtask {k:02d}] {len(keys)} key-frames {keys}, "
              f"prompts {prompts}")
        if prompt_roles:
            print(f"    roles: {dict(zip(prompts, prompt_roles))}")
        frames = [self._load_keyframe(k, t) for t in keys]
        for i, prompt in enumerate(prompts):
            role = prompt_roles[i] if prompt_roles and i < len(prompt_roles) \
                else None
            self._process_prompt(seg_dir, k, keys, frames, seg_dets,
                                 prompt, role)
```

- [ ] **Step 5: Apply the object span in `_process_prompt`**

Replace the signature (line 481–483):

```python
    def _process_prompt(self, seg_dir: str, k: int,
                        keyframes: list[int], frames: list[np.ndarray],
                        seg_dets: dict | None, prompt: str) -> None:
```

with:

```python
    def _process_prompt(self, seg_dir: str, k: int,
                        keyframes: list[int], frames: list[np.ndarray],
                        seg_dets: dict | None, prompt: str,
                        role: str | None = None) -> None:
```

Then replace (lines 492–497):

```python
        n = len(keyframes)
        h, w = frames[0].shape[:2]
        empty_reason = None
        in_mask_need = 0
        if n < 2:
            empty_reason = "insufficient keyframes"
```

with:

```python
        # Object prompts sample only the transport span: the 2nd to the
        # 2nd-to-last key-frame (gripper close .. open). The object is
        # static on the dropped boundary frames (sub-task start/end), so
        # points there would only duplicate the close/open ones. The
        # manipulator — and any prompt without a recorded role (folder
        # mode, older Step-3a JSONs) — keeps the full span. Fewer than 2
        # frames left -> "insufficient keyframes" below.
        if role == "object":
            keyframes = keyframes[1:-1]
            frames = frames[1:-1]
        n = len(keyframes)
        h, w = frames[0].shape[:2] if frames else (0, 0)
        empty_reason = None
        in_mask_need = 0
        if n < 2:
            empty_reason = "insufficient keyframes"
```

- [ ] **Step 6: Record the role and the full-segment envelope in the meta**

In the `meta` dict of `_process_prompt` (lines 581–600), add the role and make `"segment"` span the full keyframe list (not the subset — for the canonical object prompt the envelope `[start, end+1)` keeps meaning the sub-task's span):

```python
            "episode": int(self.ep_idx),
            "subtask": int(k),
            "segment": [int(min(keyframes_all)), int(max(keyframes_all)) + 1],
            "camera_key": self.cam_key,
            "prompt": prompt,
            "prompt_slug": slug,
            "role": role,
```

and define `keyframes_all` at the top of the prompt-local span block — the lines above the `if role == "object":` slice become:

```python
        keyframes_all = keyframes          # full sub-task list (meta span)
        # Object prompts sample only the transport span: ...
        if role == "object":
            keyframes = keyframes[1:-1]
            frames = frames[1:-1]
```

- [ ] **Step 7: Syntax and edge smoke checks**

Run: `python -m py_compile tools/general_test/pipeline/run_object_init_points.py`
Expected: exit 0.

Then check the degenerate cases by inspection (no run): with `role == "object"` and a 2-frame keyframe list, `keyframes = []` and `frames = []`, `n == 0`, `h, w = (0, 0)` (guarded), the SAM3 loop over `zip(keyframes, frames)` is empty, `np.savez` writes `keypoints (0, 0, 2)`, and `min(keyframes_all)` is safe (the envelope uses `keyframes_all`, which is never empty upstream — segments with no keyframes are skipped in `_process_episode`). With a 3-frame list, `n == 1` → `"insufficient keyframes"`.

- [ ] **Step 8: Commit**

```bash
git add tools/general_test/pipeline/run_object_init_points.py
git commit -m "feat(step3b): object prompts sample the close..open keyframes only

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Step 4d bounds each pass's trace window by its prompts' keyframes

**Files:**
- Modify: `src/utils/keyframe_utils.py` (after `cap_keyframes`, ~line 156)
- Modify: `tools/astribot/run_step4_traces.py` (imports line 81, module docstring lines 19–35, `parse_args` description lines 110–114, `_candidate_frames` 390–401, `_usable_at` 403–412, `_process_role` 540–542 + 559–569 + 619–620)
- Test: `tests/test_span_stems.py`

**Interfaces:**
- Consumes: Step-3b npz `frame_indices` (Task 3) — for the object role `[close, open]`, for the manipulator the full list `[start .. end]`.
- Produces: `span_stems(stems, first_frames, last_frames) -> list[int]` in `keyframe_utils`; per-pass trace steps = `span_stems(...)` from the anchor on (window's first stem where any prompt has usable keypoints). Manipulator/unlabelled passes with full-span keyframes produce exactly today's sequences.

- [ ] **Step 1: Write the failing test**

Create `tests/test_span_stems.py`:

```python
"""span_stems: the Step-2 stems of a role pass's trace window, clipped to
the prompts' Step-3 key-frame envelope."""
from utils.keyframe_utils import span_stems


def test_full_span_recovers_all_stems():
    # [start .. end] key-frames bound the whole sub-task -> every stem
    stems = list(range(0, 40, 4))  # [0, 4, ..., 36]
    assert span_stems(stems, [0], [39]) == stems


def test_object_window_off_grid():
    # stems 0, 3, 7: close key-frame 6 -> last stem <= 6 is 3; open
    # key-frame 9 -> no stem >= 9 -> clamped to the last stem (7)
    assert span_stems([0, 3, 7], [6], [9]) == [3, 7]


def test_object_window_on_grid_is_exact():
    assert span_stems([0, 3, 7], [3], [7]) == [3, 7]


def test_keyframes_before_first_stem_clamp_to_start():
    assert span_stems([4, 8, 12], [0], [10]) == [4, 8]


def test_multi_prompt_union():
    # one prompt key-framed [2, 8], another [6, 20] -> envelope [2, 20]
    stems = list(range(0, 24, 4))  # [0, 4, ..., 20]
    assert span_stems(stems, [2, 6], [8, 20]) == stems


def test_empty_inputs():
    assert span_stems([], [0], [10]) == []
    assert span_stems([0, 4], [], [10]) == [0, 4]
    assert span_stems([0, 4], [2], []) == [0, 4]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_span_stems.py -q`
Expected: FAIL — `ImportError: cannot import name 'span_stems'`.

- [ ] **Step 3: Implement `span_stems`**

In `src/utils/keyframe_utils.py`, immediately after `cap_keyframes` (ends ~line 156), add:

```python
def span_stems(stems: list[int], first_frames: list[int],
               last_frames: list[int]) -> list[int]:
    """The Step-2 stems of one role pass's trace window: the sub-list of
    ``stems`` from the last stem at-or-before the earliest first key-frame
    to the first stem at-or-after the latest last key-frame (boundaries
    inclusive; clamped to the first/last stem when the key-frames fall
    outside the grid).

    A full-span prompt (key-frames [start .. end] of the sub-task)
    therefore recovers every stem, while an object prompt key-framed only
    on the transport span ([gripper close .. open], Step 3b) is traced
    from the stem right before the close to the stem right after the open
    — the object is static on the dropped boundary key-frames, so its
    keypoints stay exact on that leading stem (see run_step4_traces.py).
    """
    if not stems or not first_frames or not last_frames:
        return list(stems)
    lo_f = min(first_frames)
    hi_f = max(last_frames)
    a = next((s for s in reversed(stems) if s <= lo_f), stems[0])
    b = next((s for s in stems if s >= hi_f), stems[-1])
    return [s for s in stems if a <= s <= b]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/test_span_stems.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Import it in the Step-4 tool**

In `tools/astribot/run_step4_traces.py`, line 81, replace:

```python
from utils.keyframe_utils import load_subtask_meta
```

with:

```python
from utils.keyframe_utils import load_subtask_meta, span_stems
```

- [ ] **Step 6: Update the module docstring and argparse description**

In the module docstring, replace the "The two roles anchor differently:" block (lines 19–35):

```
The two roles anchor differently:

- the **manipulator** keypoints are tracked from the first frame of the
  sub-task (its first Step-2 stem);
- the **object** keypoints are tracked from the sub-task's first key-frame
  that carries usable object keypoints (leading stems are skipped when the
  sub-task start has none, and the role is skipped when no key-frame of the
  sub-task has any).

A keypoint is *usable* on a key-frame when it is a surviving Step-3
keypoint lying inside that key-frame's SAM3 mask (masks[j].any() missing
-> unconstrained) with valid depth at its pixel. Roles come from the
dataset annotations meta/subtasks.csv ([object, manipulator] of the
sub-task's row): Step 3a recorded the segment's canonical sub-task label
(subtask_index) in the detections JSON, and that label resolves the row —
a segment without a recorded label is tracked unlabelled (anchored like
the object), never role-matched by the segment ordinal.
```

with:

```
Each role pass traces its prompts over the Step-2 stems inside a window
bounded by the prompts' Step-3 key-frames (span_stems): from the last
stem at-or-before the earliest first key-frame to the first stem
at-or-after the latest last key-frame.

- the **manipulator** key-frames span the whole sub-task ([start frame ..
  last frame]), so its pass tracks from the sub-task's first stem to its
  last, as before;
- the **object** key-frames span only the transport ([gripper close ..
  gripper open] — Step 3b samples between the sub-task's 2nd and 2nd-to-
  last key-frame), so its pass tracks the stems from just before the
  close to just after the open; the object is static outside that span,
  so its close-frame keypoints stay exact on the stem right before the
  close (the gripper has not occluded them yet).

A keypoint is *usable* on a key-frame when it is a surviving Step-3
keypoint lying inside that key-frame's SAM3 mask (masks[j].any() missing
-> unconstrained) with valid depth at its pixel. Roles come from the
dataset annotations meta/subtasks.csv ([object, manipulator] of the
sub-task's row): Step 3a recorded the segment's canonical sub-task label
(subtask_index) in the detections JSON, and that label resolves the row —
a segment without a recorded label is tracked unlabelled, never
role-matched by the segment ordinal.
```

Update the unlabelled-prompt log in `_process_segment` (lines 511–515) — the "anchored like the object" wording is stale now that every pass anchors by its own prompts' keyframes:

```python
            if role is None:
                print(f"    [{p['slug']}] warning: prompt {p['prompt']!r} "
                      f"matches no object/manipulator entry of sub-task {k}; "
                      f"tracking it in the unlabelled pass")
```

In `parse_args`, replace the description (lines 110–114):

```python
        description="Step 4 online: track the Step-3 keypoints with TAPIP3D "
                    "over the Step-2 depth + pose outputs — one pass per role "
                    "(object anchored at the sub-task's first usable key-frame, "
                    "manipulator at the sub-task's first frame), RGB frames "
                    "decoded online from the dataset."
```

with:

```python
        description="Step 4 online: track the Step-3 keypoints with TAPIP3D "
                    "over the Step-2 depth + pose outputs — one pass per role "
                    "(each pass traces the Step-2 stems around its prompts' "
                    "key-frames: the object's close..open transport, the "
                    "manipulator's whole sub-task), RGB frames decoded online "
                    "from the dataset."
```

- [ ] **Step 7: Bound the candidates and the usable check by the window**

Replace `_candidate_frames` (lines 390–401):

```python
    def _candidate_frames(self, prompts: list[dict]) -> list[int]:
        """Chronological anchor candidates of one role pass: the sub-task's
        first Step-2 stem first (the manipulator's rule — the object pass
        finds the same frame when its keypoints are usable there), then the
        role's key-frames that are Step-2 stems."""
        stems = self.seg_stems
        cands = [stems[0]]
        for t in sorted({int(i) for p in prompts
                         for i in p["frame_indices"]}):
            if t in stems and t not in cands:
                cands.append(t)
        return cands
```

with:

```python
    def _candidate_frames(self, prompts: list[dict],
                          window: list[int]) -> list[int]:
        """Chronological anchor candidates of one role pass inside its trace
        window: the window's first stem, then the prompts' key-frames that
        are stems of the window."""
        cands = [window[0]]
        for t in sorted({int(i) for p in prompts
                         for i in p["frame_indices"]}):
            if t in window and t != cands[0]:
                cands.append(t)
        return cands
```

Replace `_usable_at` (lines 403–412):

```python
    def _usable_at(self, prompt: dict, kf_abs: int, depth: np.ndarray):
        """(rows, px_keyframe, px_depth) of a prompt's keypoints usable on
        the key-frame (Step-2 stem) kf_abs; empty rows when the key-frame
        is not among the prompt's key-frames or nothing is usable there."""
        kfs = list(prompt["frame_indices"])
        if kf_abs not in kfs:
            return np.empty(0, dtype=np.int64), None, None
        j = kfs.index(kf_abs)
        return _row_pixels(prompt["keypoints"], prompt["masks"], j, depth,
                           MAX_OBJECT_QUERIES)
```

with:

```python
    def _usable_at(self, prompt: dict, kf_abs: int, depth: np.ndarray):
        """(rows, px_keyframe, px_depth) of a prompt's keypoints usable on
        the Step-2 stem kf_abs: the prompt's own key-frame column when the
        stem is one of its key-frames, else the first column's keypoints
        tested against kf_abs's depth — valid for the window's leading stem
        (at-or-before the first key-frame), where the object is still
        static. Empty rows when nothing is usable there."""
        kfs = list(prompt["frame_indices"])
        j = kfs.index(kf_abs) if kf_abs in kfs else 0
        return _row_pixels(prompt["keypoints"], prompt["masks"], j, depth,
                           MAX_OBJECT_QUERIES)
```

- [ ] **Step 8: Compute the window and bound the sequence in `_process_role`**

Update `_process_role`'s docstring (lines 540–542) "…over the stems from the anchor on" → "…over the trace window's stems from the anchor on".

Replace the anchor block (lines 559–569):

```python
        # --- anchor: first candidate frame with usable keypoints ------------
        depth = None
        anchor_abs = None
        usable: dict[str, tuple] = {}
        for cand in self._candidate_frames(prompts):
            depth, intrs, extr = _geometry_at(self.seg_depth_dir, cand)
            for p in prompts:
                usable[p["slug"]] = self._usable_at(p, cand, depth)
            if any(len(u[0]) for u in usable.values()):
                anchor_abs = cand
                break
        if anchor_abs is None:
            reason = "no usable keypoints on any key-frame of the " \
                     "sub-task's Step-2 steps"
            print(f"  [subtask {k:02d}] {label} pass: empty ({reason})")
            for p, d in zip(prompts, pdirs):
                self._save_empty(p, role, d, reason, seg_report)
            return
```

with:

```python
        # --- trace window + anchor: first candidate with usable rows --------
        # The pass traces the stems inside its prompts' key-frame envelope
        # (span_stems): the object's [close .. open] transport gets the stem
        # right before the close through the stem right after the open; a
        # full-span prompt keeps the whole sub-task (as before).
        window = span_stems(
            self.seg_stems,
            [int(p["frame_indices"][0]) for p in prompts],
            [int(p["frame_indices"][-1]) for p in prompts])
        depth = None
        anchor_abs = None
        usable: dict[str, tuple] = {}
        for cand in self._candidate_frames(prompts, window):
            depth, intrs, extr = _geometry_at(self.seg_depth_dir, cand)
            for p in prompts:
                usable[p["slug"]] = self._usable_at(p, cand, depth)
            if any(len(u[0]) for u in usable.values()):
                anchor_abs = cand
                break
        if anchor_abs is None:
            reason = "no usable keypoints on the pass's candidate frames " \
                     f"(window {window[0]}..{window[-1]})"
            print(f"  [subtask {k:02d}] {label} pass: empty ({reason})")
            for p, d in zip(prompts, pdirs):
                self._save_empty(p, role, d, reason, seg_report)
            return
```

Replace the sequence line (619–620):

```python
        # --- sequence: Step-2 stems from the anchor on ------------------------
        steps = self.seg_stems[self.seg_stems.index(anchor_abs):]
```

with:

```python
        # --- sequence: the trace window's stems from the anchor on ------------
        steps = window[window.index(anchor_abs):]
```

(Note: `_process_segment`'s early `skip-done` / `_save_empty` bookkeeping is untouched; `anchor_abs` is a member of `window` by construction, since candidates are drawn from it.)

- [ ] **Step 9: Syntax and unit checks**

Run: `python -m py_compile tools/astribot/run_step4_traces.py`
Expected: exit 0.

Run: `uv run --extra dev pytest tests/ -q`
Expected: PASS (test_image_io + the new test files; no failures).

- [ ] **Step 10: Commit**

```bash
git add src/utils/keyframe_utils.py tests/test_span_stems.py tools/astribot/run_step4_traces.py
git commit -m "feat(step4): bound each pass trace window by its prompts' keyframes

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: README and docs span wording

**Files:**
- Modify: `README.md`, `docs/astribot/astribot_traces.md`, `docs/general_test/general_test.md`, `docs/general_test/pipeline/step3.md`, `docs/general_test/module/romav2.md`, `docs/general_test/module/rexomni.md`

Apply the following edits (line numbers refer to the current files — re-locate by the quoted text if the file shifted):

**README.md**

1. In "### Step 3 — Sampling Keypoints", item 3 (currently "…to find matching keypoints consistent across the key-frames of a subtask."), append a new sentence after that item's text:

```
The **manipulator** crops are matched over all key-frames of the subtask
(start frame → last frame); the **object** crops only over the sub-task's
2nd to 2nd-to-last key-frame — the gripper close/open pair that carries
the object. The object is static before the close and after the open, so
matching on the boundary frames would only duplicate the close/open
points.
```

2. In "### Step 4 — 3D Trace", replace the paragraph "We use **TAPIP3D** to track the 3D positions of the keypoints detected in Step 3, from the first frame to the last frame of each subtask, and save the output." with:

```
We use **TAPIP3D** to track the 3D positions of the keypoints detected in
Step 3, and save the output. The **manipulator** keypoints are tracked
from the subtask's first frame to its last; the **object** keypoints only
across its transport — from the frame just before the gripper-close
key-frame to the frame just after the gripper-open key-frame (outside
that span the object is static). Tracking runs on the Step-2 depth + pose
frames.
```

3. In the Case-2 usage comment block, replace the Step-3 comment lines with:

```
# Step 3 — key-point sampling: key-frame jpgs + RexOmni detections +
# SAM3/RoMaV2 init points (prompts come from the dataset's meta/subtasks.csv;
# object keypoints sample between the gripper close/open key-frames, the
# manipulator over all key-frames)
```

and the Step-4 comment lines with:

```
# Step 4 — 3D traces: track the Step-3 keypoints with TAPIP3D over the
# Step-2 depth + pose outputs (one pass per role: the object keypoints
# over its close..open transport, the manipulator over the whole
# sub-task) -> eps_data/traces/
```

**docs/general_test/general_test.md**

This page documents the Case-1 folder flow (`--text-prompts`, no roles) and its Step 4 is the standalone `infer_tapip3d.py` demo — the role split does not apply there. Only the Step-3 bullet gets a dataset-mode qualifier; the Step-4 bullet stays untouched.

- In the Step-3 bullet (lines ~73–81), after "…and only the top-k keypoints inside the masks are kept." insert: "In dataset (episode) mode, where each prompt carries its role, object prompts are matched between the sub-task's gripper close/open key-frames (2nd..2nd-to-last) and the manipulator over all key-frames; folder mode (`--text-prompts`, no roles) matches over all key-frames as before."

**docs/general_test/pipeline/step3.md**

This doc covers both modes (dataset + folder). Qualify the split statements as dataset-mode; the folder-mode RoMAv2 stanza keeps its generic wording.

- Intro (lines 3–6): after "…keep the top-k keypoints inside the masks." append: "In dataset mode, object prompts are matched between the sub-task's 2nd and 2nd-to-last key-frame (the gripper close/open pair); manipulator prompts — and all folder-mode prompts, which carry no role — over all key-frames."
- Flow diagram (lines 24–26): change the RoMAv2 stanza to mention the split:

```
  → RoMAv2      — match keypoints across key-frames on enlarged
                  bbox crops (mask cropped with the same box, so
                  points are sampled inside the object only;
                  dataset mode: object prompts span the close..open
                  key-frames)
```

- The checklist item describing the Step-3a JSON (lines ~136–140): extend "…and the `prompts` of that label's row in `meta/subtasks.csv` ([object, manipulator] — …)" with "…together with the aligned `prompt_roles` ([object, manipulator] — …)" and mention "folder mode records no roles".
- The `init_points.npz` checklist item (lines 143–144): replace "N = number of key-frames" with "N = number of matched key-frames (dataset mode: 2 for an object prompt of a canonical 4-frame sub-task — the close/open pair — all key-frames otherwise)".

**docs/general_test/module/romav2.md**

- Lines 81–87 ("Usage in the pipeline"): replace "…finds the keypoints that match consistently **across the key-frames** of the subtask." with "…finds the keypoints that match consistently **across the key-frames** of the subtask (dataset mode: an object prompt is matched across its gripper close/open key-frames only — the sub-task's 2nd..2nd-to-last — while the manipulator spans all of them; folder mode, which records no roles, always spans all of them)." and replace the closing "…TAPIP3D then tracks them over the whole subtask (Step 4, see [`tapip3d.md`](tapip3d.md))." with "…TAPIP3D then tracks them (Step 4, see [`tapip3d.md`](tapip3d.md)): the manipulator over the whole subtask, the object over its close/open transport."

**docs/general_test/module/rexomni.md**

- Lines 124–128: after "…drive the detection on the key-frames (start / end frames included)." append: "In dataset mode each prompt is recorded in the detections JSON with its role (`prompt_roles` — the csv column it was read from), which Step 3b uses to sample object keypoints only between the close/open key-frames."

**docs/astribot/astribot_traces.md**

1. Replace the role/anchor paragraph and table (lines 21–30):

```
**Two roles, two passes.** The sub-task's prompts are split by role —
matching the prompt text to the [object, manipulator] entries of the
sub-task's row in the dataset's `meta/subtasks.csv` (the same source Step
3a/3b use). Each role is tracked in its **own independent TAPIP3D pass**.
Step 3b samples the **object** keypoints only between the sub-task's 2nd
and 2nd-to-last key-frame (the gripper close/open pair that carries the
object), and each pass traces the Step-2 stems inside its prompts'
key-frame envelope:

| Role | Step-3 key-frames | Tracked sequence (Step-2 stems) |
|---|---|---|
| **manipulator** (e.g. the gripper) | all (sub-task start → last frame) | sub-task's first stem → last stem |
| **object** (e.g. the cup) | close → open (2nd → 2nd-to-last) | last stem ≤ close → first stem ≥ open |

The window rule is role-agnostic (`span_stems` in `utils/keyframe_utils.py`):
from the last stem at-or-before the earliest first key-frame to the first
stem at-or-after the latest last key-frame. The object therefore starts on
the stem **right before the close key-frame** — the object is static until
the close, so its close-frame pixels are exact there (and the gripper has
not occluded them yet) — and ends on the stem **right after the open**.
```

2. In "**Camera and split alignment.**" (lines 62–72), replace the two sentences "A key-frame can only anchor a pass when it is a Step-2 stem (has geometry): the anchor scan runs over the role's key-frames that are stems, starting from the sub-task's first stem. The tracked sequence is *whatever Step-2 stems are saved* from the anchor on — if Step 2 is later modified to stream other per-sub-task frame ranges (e.g. at the key-frame indices), this tool follows the saved stems automatically." with:

```
A pass traces only Step-2 stems (geometry exists there). Its window is
the stems inside its prompts' key-frame envelope — the last stem
at-or-before the earliest first key-frame through the first stem
at-or-after the latest last key-frame (`span_stems`) — and the anchor is
the window's first stem where a prompt has usable keypoints. If Step 2 is
later modified to stream other per-sub-task frame ranges (e.g. at the
key-frame indices), this tool follows the saved stems automatically.
```

3. In "## Output" (lines 108–110), replace "`T` = the number of tracked stems (the sub-task's Step-2 steps from the anchor on, absolute indices listed in the prompt `metadata.json` under `steps`)" with "`T` = the number of tracked stems (the trace window's Step-2 stems from the anchor on, absolute indices listed in the prompt `metadata.json` under `steps`)".

4. In the "A keypoint is *usable*…" paragraph after the table (lines ~36–38), replace "Prompts whose text matches neither annotation column are tracked too (warned, anchored like the object)." with "Prompts whose text matches neither annotation column are tracked too (warned), in a separate unlabelled pass over their own key-frames."

5. In "## Notes", replace the bullet "**Anchor key-frames must be Step-2 stems.** … a later stem does)." (lines 149–154) with:

```
- **Anchors sit on the stem grid; boundary key-frames usually don't.**
  Step-2 streams at `--stride` 4, so a prompt's key-frames are generally
  *not* stems — except the sub-task's first frame (a boundary key-frame),
  which is always a stem. The manipulator pass therefore anchors on the
  sub-task's first stem as before; the object pass anchors on the stem
  right before the close key-frame, where the object is still static (its
  close-frame pixels are exact there, and the approach has not occluded
  them). When that stem carries no usable keypoints the anchor advances to
  the next window stem that does.
```

- [ ] **Step 1: Apply the edits above to the six files**

Verify each old string exists before replacing; adjust for drift. Do not reword passages the scan did not flag (e.g. `docs/astribot/astribot_extract_frames.md` — keyframe *definition* is unchanged).

- [ ] **Step 2: Commit**

```bash
git add README.md docs/astribot/astribot_traces.md docs/general_test/general_test.md \
        docs/general_test/pipeline/step3.md docs/general_test/module/romav2.md \
        docs/general_test/module/rexomni.md
git commit -m "docs: per-role sampling and trace spans (object close..open, manipulator full)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: End-to-end verification on a real episode

**Files:**
- Modify (only if the re-run contradicts it): `docs/astribot/astribot_traces.md` "Verified on `astri_making_coffee`…" note (lines 161–175)
- No code files.

**Interfaces:**
- Consumes Tasks 1–4. Requires, for the chosen episode: Step-1 key-frames with `subtask_labels.json`, Step-2 depth+pose for the tracked camera (cam_head = dataset camera index 0), and the dataset locally under `/data/astribot_making_coffee_vlva_full`.

- [ ] **Step 1: Check preconditions and pick the episode**

```bash
ls /data/astribot_making_coffee_vlva_full/eps_data/key_frames/           # ep dirs
ls /data/astribot_making_coffee_vlva_full/eps_data/key_frames/ep000000/  # subtask_labels.json?
ls /data/astribot_making_coffee_vlva_full/eps_data/depth_pose/ep000000/subtask_00/depth_cam_head/ | head
```

Choose the first episode with key-frames + a `depth_cam_head` Step-2 folder (start with `ep000000`). If no episode has Step-2 outputs for `cam_head`, run Step 2 first for episode 0 (GPU streaming, main env):

```bash
python tools/astribot/run_step2_depth_stream.py \
    --repo-id Kronze157/astribot_making_coffee_vlva_full \
    --data-root /data/astribot_making_coffee_vlva_full \
    --episode-idxes 0 --backend vggt_omega --camera-idxes 0
```

If key-frames are missing entirely, run Step 1 first (per `scripts/astribot/extract_frames.sh`), then Step 2 above. Record the chosen `EP`.

- [ ] **Step 2: Re-run Step 3a and inspect the recorded roles**

```bash
.venv-rexomni/bin/python tools/general_test/pipeline/run_object_detection.py \
    --data-root /data/astribot_making_coffee_vlva_full --episode-idxes "$EP"
python - <<'PY'
import json
p = f"/data/astribot_making_coffee_vlva_full/eps_data/detections/ep{int('$EP'):06d}.json"
d = json.load(open(p))
for k, s in sorted(d["subtasks"].items()):
    print(k, s.get("prompts"), s.get("prompt_roles"))
PY
```

Expected: every subtask with prompts prints aligned `prompt_roles` of `["object", "manipulator"]` (or a subset for rows with empty cells); print lines and the JSON agree. Pick one subtask whose printed `keyframes` list has length 4 (the canonical `[start, close, open, end]`) and note its `k` and indices for Step 3.

- [ ] **Step 3: Re-run Step 3b and inspect the object/manipulator spans**

```bash
python tools/general_test/pipeline/run_object_init_points.py \
    --data-root /data/astribot_making_coffee_vlva_full --episode-idxes "$EP"
python - <<'PY'
import json, numpy as np, os
base = f"/data/astribot_making_coffee_vlva_full/eps_data/init_points/ep{int('$EP'):06d}"
for sub in sorted(os.listdir(base)):
    for slug in sorted(os.listdir(f"{base}/{sub}")):
        npz = np.load(f"{base}/{sub}/{slug}/init_points.npz")
        meta = json.load(open(f"{base}/{sub}/{slug}/init_points.json"))
        print(sub, slug, meta.get("role"),
              "frame_indices", npz["frame_indices"].tolist(),
              "K", npz["keypoints"].shape[0], "empty", meta.get("empty_reason"))
PY
```

Expected: for each canonical subtask, the **object** prompt's npz lists exactly its 2 interior key-frames (`[close, open]` — the folder's frame list minus first and last) with `role == "object"`, and the **manipulator** prompt lists all of them (`role == "manipulator"`). Both npz files are non-empty (`K > 0`) and `viz.png` exists for both. A subtask whose keyframes are < 4 long reports the object prompt as `empty_reason: "insufficient keyframes"` — that is the accepted consequence of the "always `[1:-1]`" rule, not an error.

- [ ] **Step 4: Re-run Step 4 and inspect the per-role trace windows**

```bash
python tools/astribot/run_step4_traces.py \
    --repo-id Kronze157/astribot_making_coffee_vlva_full \
    --data-root /data/astribot_making_coffee_vlva_full --episode-idxes "$EP"
python - <<'PY'
import json, numpy as np, os, glob
def stems_of(sub):
    d = glob.glob(f"/data/astribot_making_coffee_vlva_full/eps_data/depth_pose/ep{int('$EP'):06d}/{sub}/depth_cam_head/*.npz")
    return sorted(int(os.path.basename(p).split("_")[1].split(".")[0]) for p in d)
base = f"/data/astribot_making_coffee_vlva_full/eps_data/traces/ep{int('$EP'):06d}"
for sub in sorted(os.listdir(base)):
    stems = stems_of(sub)
    for slug in sorted(os.listdir(f"{base}/{sub}")):
        m = json.load(open(f"{base}/{sub}/{slug}/metadata.json"))
        if m.get("status") != "ok":
            print(sub, slug, m.get("status"), m.get("empty_reason")); continue
        steps = m["steps"]
        print(sub, slug, m["role"], "anchor", m["anchor_frame"],
              "steps", steps[0], "..", steps[-1], f"({len(steps)})",
              "stems", stems[0], "..", stems[-1])
PY
```

Expected, for the chosen canonical subtask (npz frame indices `[close, open]`, folder `[start, close, open, end]`):
- object pass: `anchor == max(stem ≤ close)`, `steps[-1] == min(stem ≥ open)` (clamped to the last stem when the open frame lies beyond the grid), and `steps` are exactly the stems in that range — do **not** run to the sub-task's last stem.
- manipulator pass: `steps` run from `stems[0]` to `stems[-1]` (identical to a pre-change run: first stem → last stem).
- Both passes tracked `> 0` role keypoints (`status: "ok"`).

Cross-check one trace on the object metadata: `steps[0]` is the frame where the object is still at rest (before the close key-frame) and `coords.npy` row 0 ≈ the unprojected query points.

- [ ] **Step 5: Full test suite + amend the verification note if contradicted**

Run: `uv run --extra dev pytest tests/ -q`
Expected: PASS (all test files).

If the re-run numbers contradict the "Verified on `astri_making_coffee`…" note in `docs/astribot/astribot_traces.md` (it predates the object-window split and states both roles anchored at the sub-task's first stem), update the note to reflect the new run — e.g. append a dated sentence: "Re-verified 2026-09-06 after the object/manipulator span split: object passes now anchor on the stem right before the close key-frame and end at the stem right after the open one; the manipulator passes are unchanged." Otherwise leave it.

- [ ] **Step 6: Commit (only if Step 5 edited the note)**

```bash
git add docs/astribot/astribot_traces.md
git commit -m "docs(step4): re-verify trace note after the object-window split

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review Notes (run after writing — see plan header)

- **Spec coverage:** requirements (1)–(4) map to Task 3 (object `[1:-1]`, empty consequence), Task 2+3 (roles on disk → full-span fallback), Task 4 (stem-before-close → stem-after-open, no Step-2 change), Task 4 (manipulator unchanged via full-span envelope). Docs → Task 5; real-data proof → Task 6.
- **Type consistency:** `prompt_roles` aligned lists recorded in Task 2 (episode mode), consumed by Task 3; `frame_indices` npz arrays produced in Task 3 feed `span_stems(first_frames=…[0], last_frames=…[-1])` in Task 4; `role == "object"` literal matches the csv column names recorded by Task 2 (`SUBTASK_PROMPT_COLUMNS`).
- **Placeholders:** none — every edit carries its replacement code/text.
