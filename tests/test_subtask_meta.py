"""CPU-only unit tests for the Step-3 subtask-annotation loader
(src/utils/keyframe_utils.py): load_subtask_meta / subtask_prompts."""

from __future__ import annotations

import pytest

from utils.keyframe_utils import (
    load_subtask_meta,
    subtask_prompts,
)

#: The on-disk shape of meta/subtasks.csv is hand-edited in a spreadsheet,
#: so headers keep leading spaces, values trail whitespace, an unnamed
#: trailing column exists, and one stray cell lands under it.
MESSY_CSV = (
    'subtask_index,subtask," manipulator"," object"," destination",\n'
    '0,use the left hand to pick the brown cup,'
    '" left robot arm\'s grippers"," brown cup",'
    '" drip tray",\n'
    '1,use the right hand to press the button,'
    '" right robot arm\'s grippers"," middle button",\n'
    '2,use the right hand to place the cup onto the tray,'
    '," brown cup"," transparent tray","the middle of the table"\n'
)


@pytest.fixture()
def data_root(tmp_path):
    """Dataset root with a messy meta/subtasks.csv mirroring the real file."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "subtasks.csv").write_text(MESSY_CSV)
    return tmp_path


def test_load_subtask_meta_normalizes_headers_and_cells(data_root):
    meta = load_subtask_meta(data_root)
    assert sorted(meta) == [0, 1, 2]
    # headers stripped of their spreadsheet spaces, values stripped
    assert meta[0]["manipulator"] == "left robot arm's grippers"
    assert meta[0]["object"] == "brown cup"
    assert meta[0]["subtask"] == "use the left hand to pick the brown cup"
    # unnamed trailing column dropped (including its stray cell)
    assert set(meta[2]) == {"subtask", "object", "destination"}
    assert meta[2]["destination"] == "transparent tray"
    # row with an empty manipulator cell has no such key
    assert "manipulator" not in meta[2]
    # a row shorter than the header list still parses
    assert meta[1]["object"] == "middle button"
    # empty trailing cell = empty destination column, dropped
    assert "destination" not in meta[1]


def test_load_subtask_meta_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="subtasks.csv"):
        load_subtask_meta(tmp_path)


def test_load_subtask_meta_header_only(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "subtasks.csv").write_text(
        'subtask_index,subtask," manipulator"," object"\n')
    assert load_subtask_meta(tmp_path) == {}


def test_load_subtask_meta_skips_non_integer_index_rows(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "subtasks.csv").write_text(
        "subtask_index,subtask,manipulator,object\n"
        "0,pick the cup,left gripper,brown cup\n"
        ",row without index,left gripper,brown cup\n"
        "not-a-number,broken,left gripper,brown cup\n")
    assert load_subtask_meta(tmp_path) == {
        0: {"subtask": "pick the cup", "manipulator": "left gripper",
            "object": "brown cup"}
    }


def test_load_subtask_meta_no_index_column(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "subtasks.csv").write_text(
        'subtask,manipulator,object\npick the cup,left gripper,brown cup\n')
    with pytest.raises(ValueError, match="subtask_index"):
        load_subtask_meta(tmp_path)


def test_subtask_prompts_order_object_then_manipulator():
    row = {"object": "brown cup", "manipulator": "left robot arm's grippers"}
    assert subtask_prompts(row) == ["brown cup",
                                    "left robot arm's grippers"]


def test_subtask_prompts_skip_empty_and_missing_columns():
    assert subtask_prompts({"object": "brown cup"}) == ["brown cup"]
    assert subtask_prompts({"manipulator": "left gripper"}) == \
        ["left gripper"]
    assert subtask_prompts({}) == []
    assert subtask_prompts(None) == []
