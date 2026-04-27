def test_public_imports():
    from langtrans import (
        ActionNode,
        Proc,
        Spec,
        SwitchNode,
        Trans,
        action,
        concurrent,
        loop,
        optional,
        sequential,
        switch,
    )

    assert Trans is not None
    assert Proc is not None
    assert action is not None
    assert Spec is not None
    assert ActionNode is not None
    assert SwitchNode is not None
    assert sequential is not None
    assert concurrent is not None
    assert optional is not None
    assert loop is not None
    assert switch is not None
