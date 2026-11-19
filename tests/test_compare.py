from src.compare import MISSED, _segments, _text, edit_distance

def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0

def test_text_keeps_missed_marker_and_applies_backspace():
    assert _text(["a", MISSED, "x", "backspace", "space", "b"]) == f"a{MISSED} b"

def test_segments_split_after_enter():
    assert _segments(["a", "enter", "b", "c"]) == [[0, 1], [2, 3]]
