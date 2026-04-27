from langtrans.spec import Spec


def test_spec_from_function():
    s = Spec(lambda state: state["metadata"].get("x") == 1)
    assert s({"metadata": {"x": 1}}) is True
    assert s({"metadata": {"x": 2}}) is False


def test_spec_and():
    a = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    b = Spec(lambda s: s["metadata"].get("y", 0) > 0)
    combined = a & b
    assert combined({"metadata": {"x": 1, "y": 1}}) is True
    assert combined({"metadata": {"x": 1, "y": 0}}) is False
    assert combined({"metadata": {"x": 0, "y": 1}}) is False


def test_spec_or():
    a = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    b = Spec(lambda s: s["metadata"].get("y", 0) > 0)
    combined = a | b
    assert combined({"metadata": {"x": 1, "y": 0}}) is True
    assert combined({"metadata": {"x": 0, "y": 1}}) is True
    assert combined({"metadata": {"x": 0, "y": 0}}) is False


def test_spec_not():
    a = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    inverted = ~a
    assert inverted({"metadata": {"x": 0}}) is True
    assert inverted({"metadata": {"x": 1}}) is False


def test_spec_complex_composition():
    x_pos = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    y_pos = Spec(lambda s: s["metadata"].get("y", 0) > 0)
    z_neg = Spec(lambda s: s["metadata"].get("z", 0) < 0)
    combined = (x_pos & y_pos) | ~z_neg
    assert combined({"metadata": {"x": 1, "y": 1, "z": 5}}) is True
    assert combined({"metadata": {"x": 0, "y": 0, "z": 5}}) is True
    assert combined({"metadata": {"x": 0, "y": 0, "z": -1}}) is False
