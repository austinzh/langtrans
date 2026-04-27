from langtrans.nodes import (
    ActionNode, ConcurrentNode, LoopNode, Node, OptionalNode,
    SequentialNode,
)

def dummy(state):
    return state

def dummy_rollback(state):
    return state

def dummy_guard(state) -> bool:
    return True

class TestActionNode:
    def test_minimal(self):
        node = ActionNode(func=dummy)
        assert node.func is dummy
        assert node.rollback is None
        assert node.name is None

    def test_with_rollback_and_name(self):
        node = ActionNode(func=dummy, rollback=dummy_rollback, name="my_action")
        assert node.rollback is dummy_rollback
        assert node.name == "my_action"

class TestSequentialNode:
    def test_children(self):
        a = ActionNode(func=dummy)
        b = ActionNode(func=dummy)
        node = SequentialNode(children=[a, b])
        assert len(node.children) == 2
        assert node.children[0] is a

class TestConcurrentNode:
    def test_children(self):
        a = ActionNode(func=dummy)
        b = ActionNode(func=dummy)
        node = ConcurrentNode(children=[a, b])
        assert len(node.children) == 2

class TestOptionalNode:
    def test_with_both_branches(self):
        a = ActionNode(func=dummy)
        b = ActionNode(func=dummy)
        node = OptionalNode(guard=dummy_guard, then_=a, else_=b)
        assert node.guard is dummy_guard
        assert node.then_ is a
        assert node.else_ is b

    def test_without_else(self):
        a = ActionNode(func=dummy)
        node = OptionalNode(guard=dummy_guard, then_=a)
        assert node.else_ is None

class TestLoopNode:
    def test_fixed_count(self):
        body = ActionNode(func=dummy)
        node = LoopNode(body=body, times=3)
        assert node.times == 3
        assert node.until is None

    def test_condition_based(self):
        body = ActionNode(func=dummy)
        node = LoopNode(body=body, until=dummy_guard)
        assert node.until is dummy_guard
        assert node.times is None

class TestNamedNodes:
    def test_sequential_with_name(self):
        node = SequentialNode(children=[ActionNode(func=dummy)], name="sub_flow")
        assert node.name == "sub_flow"

    def test_sequential_without_name(self):
        node = SequentialNode(children=[ActionNode(func=dummy)])
        assert node.name is None

class TestNodeUnionType:
    def test_action_is_node(self):
        node: Node = ActionNode(func=dummy)
        assert isinstance(node, ActionNode)

    def test_sequential_is_node(self):
        node: Node = SequentialNode(children=[])
        assert isinstance(node, SequentialNode)
