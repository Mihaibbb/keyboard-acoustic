import csv
from src.record import KeyTracker, write_keyup_csv

def test_autorepeat_is_one_press_and_one_release():
    k = KeyTracker()
    assert k.press(1, "a", 0.10)
    assert not k.press(1, "a", 0.15)      # autorepeat OS
    assert k.release(1, 0.30)
    assert k.presses == [(0.10, "a")]
    assert k.releases == [(0.30, "a", 0.10)]

def test_shift_released_first_keeps_capital_name():
    # Shift↓ A↓ Shift↑ A↑: la eliberare pynput ar da „a”, dar e aceeași apăsare „A”.
    k = KeyTracker()
    k.press("shift", "shift", 1.00)
    k.press(0, "A", 1.10)
    k.release("shift", 1.20)
    k.release(0, 1.25)
    assert k.releases == [(1.20, "shift", 1.00), (1.25, "A", 1.10)]

def test_letter_released_first_and_shift_held_over_several_letters():
    # Shift↓ A↓ A↑ B↓ B↑ Shift↑ (cuvânt cu majuscule, Shift ținut)
    k = KeyTracker()
    k.press("shift", "shift", 0.0)
    k.press(0, "A", 0.1); k.release(0, 0.15)
    k.press(11, "B", 0.2); k.release(11, 0.25)
    k.release("shift", 0.4)
    assert [n for _, n in k.presses] == ["shift", "A", "B"]
    assert k.releases[-1] == (0.4, "shift", 0.0)

def test_release_without_press_is_ignored():
    k = KeyTracker()
    assert not k.release(5, 0.2)          # apăsată înainte de pornire
    assert k.releases == []

def test_write_keyup_csv_sorted(tmp_path):
    p = tmp_path / "keyup.csv"
    write_keyup_csv(p, [(0.5, "b", 0.4), (0.2, "a", 0.1)])
    rows = list(csv.DictReader(open(p)))
    assert rows == [{"timestamp_sec": "0.2000", "key": "a", "press_sec": "0.1000"},
                    {"timestamp_sec": "0.5000", "key": "b", "press_sec": "0.4000"}]
