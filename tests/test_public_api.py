def test_public_imports():
    from langtrans import Trans, action, Spec
    from langtrans import (
        ActionNode, SequentialNode, ConcurrentNode, OptionalNode,
        LoopNode, ProcedureNode, SwitchNode, Node,
    )
    assert Trans is not None
    assert action is not None
    assert Spec is not None
    assert ActionNode is not None
    assert SwitchNode is not None
