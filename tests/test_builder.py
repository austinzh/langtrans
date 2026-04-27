from langtrans.builder import Proc, action
from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    OptionalNode,
    SequentialNode,
    SwitchNode,
)
from langtrans.spec import Spec


def dummy(state):
    return state


def dummy_rollback(state):
    return state


def guard_true(state) -> bool:
    return True


class TestActionDecorator:
    def test_bare_decorator(self):
        @action
        def my_func(state):
            return state

        assert hasattr(my_func, "_langtrans_rollback")
        assert my_func._langtrans_rollback is None
        assert my_func("test") == "test"

    def test_decorator_with_rollback(self):
        def rollback_fn(state):
            return state

        @action(rollback=rollback_fn)
        def my_func(state):
            return state

        assert my_func._langtrans_rollback is rollback_fn
        assert my_func("test") == "test"


class TestTransBuilderSinglePrimitive:
    def test_sequential(self):
        tree = Proc().sequential(dummy, dummy).build()
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 2
        assert all(isinstance(c, ActionNode) for c in tree.children)

    def test_concurrent(self):
        tree = Proc().concurrent(dummy, dummy, dummy).build()
        assert isinstance(tree, ConcurrentNode)
        assert len(tree.children) == 3

    def test_optional_both_branches(self):
        tree = Proc().optional(guard_true, then_=dummy, else_=dummy).build()
        assert isinstance(tree, OptionalNode)
        assert isinstance(tree.then_, ActionNode)
        assert isinstance(tree.else_, ActionNode)

    def test_optional_then_only(self):
        tree = Proc().optional(guard_true, then_=dummy).build()
        assert isinstance(tree, OptionalNode)
        assert tree.else_ is None

    def test_loop_fixed_count(self):
        tree = Proc().loop(body=dummy, times=3).build()
        assert isinstance(tree, LoopNode)
        assert tree.times == 3
        assert isinstance(tree.body, ActionNode)

    def test_loop_until_condition(self):
        tree = Proc().loop(body=dummy, until=guard_true).build()
        assert isinstance(tree, LoopNode)
        assert tree.until is guard_true

    def test_procedure(self):
        tree = Proc("sub_flow").sequential(dummy, dummy).build()
        assert isinstance(tree, SequentialNode)
        assert tree.name == "sub_flow"
        assert len(tree.children) == 2


class TestTransBuilderChaining:
    def test_two_methods_wrap_in_sequential(self):
        tree = Proc().sequential(dummy, dummy).concurrent(dummy, dummy).build()
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 2
        assert isinstance(tree.children[0], SequentialNode)
        assert isinstance(tree.children[1], ConcurrentNode)

    def test_single_method_no_wrapping(self):
        tree = Proc().sequential(dummy, dummy).build()
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 2
        assert all(isinstance(c, ActionNode) for c in tree.children)


class TestTransBuilderNesting:
    def test_nested_trans_as_argument(self):
        tree = (
            Proc()
            .concurrent(
                dummy,
                Proc().sequential(dummy, dummy),
            )
            .build()
        )
        assert isinstance(tree, ConcurrentNode)
        assert len(tree.children) == 2
        assert isinstance(tree.children[0], ActionNode)
        assert isinstance(tree.children[1], SequentialNode)

    def test_deeply_nested(self):
        tree = (
            Proc()
            .sequential(
                dummy,
                Proc().concurrent(
                    Proc().sequential(dummy, dummy),
                    dummy,
                ),
                Proc().optional(guard_true, then_=dummy, else_=dummy),
            )
            .build()
        )
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 3
        assert isinstance(tree.children[0], ActionNode)
        assert isinstance(tree.children[1], ConcurrentNode)
        assert isinstance(tree.children[2], OptionalNode)


class TestActionDecoratorInBuilder:
    def test_action_with_rollback_preserves_rollback(self):
        @action(rollback=dummy_rollback)
        def my_action(state):
            return state

        tree = Proc().sequential(my_action).build()
        assert isinstance(tree, SequentialNode)
        action_node = tree.children[0]
        assert isinstance(action_node, ActionNode)
        assert action_node.rollback is dummy_rollback

    def test_spec_guard_in_optional(self):
        spec = Spec(guard_true)
        tree = Proc().optional(spec, then_=dummy).build()
        assert isinstance(tree, OptionalNode)
        assert isinstance(tree.guard, Spec)


class TestNamedPrimitives:
    def test_named_sequential(self):
        tree = Proc().sequential(dummy, dummy, name="fetch_phase").build()
        assert isinstance(tree, SequentialNode)
        assert tree.name == "fetch_phase"

    def test_named_concurrent(self):
        tree = Proc().concurrent(dummy, dummy, name="parallel_search").build()
        assert isinstance(tree, ConcurrentNode)
        assert tree.name == "parallel_search"

    def test_named_optional(self):
        tree = Proc().optional(guard_true, then_=dummy, name="routing").build()
        assert isinstance(tree, OptionalNode)
        assert tree.name == "routing"

    def test_named_loop(self):
        tree = Proc().loop(body=dummy, times=3, name="retry_loop").build()
        assert isinstance(tree, LoopNode)
        assert tree.name == "retry_loop"

    def test_named_switch(self):
        tree = (
            Proc()
            .switch(
                key=lambda s: "a", cases={"a": dummy, "b": dummy}, name="dispatcher"
            )
            .build()
        )
        assert isinstance(tree, SwitchNode)
        assert tree.name == "dispatcher"

    def test_unnamed_no_wrapping(self):
        tree = Proc().sequential(dummy, dummy).build()
        assert isinstance(tree, SequentialNode)
        assert tree.name is None
