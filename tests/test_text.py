import csv
from src.text import final_chars, final_lines, write_final_text

def test_backspace_enter_and_shifted_chars():
    events = [(0.1, "A"), (0.2, "b"), (0.3, "x"), (0.4, "backspace"),
              (0.5, "space"), (0.6, "shift"), (0.7, "!"), (0.8, "enter"),
              (0.9, "c")]
    chars, unmodeled = final_chars(events)
    assert "".join(c for _, c in chars) == "Ab !\nc"
    assert unmodeled == 0

def test_backspace_on_empty_and_unmodeled_keys():
    chars, unmodeled = final_chars([(0.1, "backspace"), (0.2, "cmd"),
                                    (0.3, "left"), (0.4, "a")])
    assert [c for _, c in chars] == ["a"]
    assert unmodeled == 2

def test_lines_keep_empty_lines_and_times():
    chars, _ = final_chars([(0.1, "enter"), (0.2, "h"), (0.3, "i"),
                            (0.4, "enter"), (0.5, "o")])
    assert final_lines(chars) == [(1, None, None, ""), (2, 0.2, 0.3, "hi"),
                                  (3, 0.5, 0.5, "o")]

def test_write_final_text_csv(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("timestamp_sec,key\n0.1,a\n0.2,comma\n0.3,\",\"\n"
                      "0.4,space\n0.5,b\n")
    text, _ = write_final_text(labels, tmp_path / "final_text.csv")
    assert text == "a, b"
    rows = list(csv.DictReader(open(tmp_path / "final_text.csv")))
    assert rows == [{"line": "1", "start_sec": "0.1000",
                     "end_sec": "0.5000", "text": "a, b"}]
