def test_public_imports():
    from langtrans import (
        ActionNode,
        Spec,
        SwitchNode,
        Trans,
        action,
    )

    assert Trans is not None
    assert action is not None
    assert Spec is not None
    assert ActionNode is not None
    assert SwitchNode is not None
