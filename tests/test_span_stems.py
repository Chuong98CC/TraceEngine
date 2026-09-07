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
    # close key-frame 0 falls before the first stem -> the window starts
    # at the first stem (4); open key-frame 10 is off the grid, so the
    # window ends at the first stem at-or-after it (12)
    assert span_stems([4, 8, 12], [0], [10]) == [4, 8, 12]


def test_multi_prompt_union():
    # one prompt key-framed [2, 8], another [6, 20] -> envelope [2, 20]
    stems = list(range(0, 24, 4))  # [0, 4, ..., 20]
    assert span_stems(stems, [2, 6], [8, 20]) == stems


def test_empty_inputs():
    assert span_stems([], [0], [10]) == []
    assert span_stems([0, 4], [], [10]) == [0, 4]
    assert span_stems([0, 4], [2], []) == [0, 4]
