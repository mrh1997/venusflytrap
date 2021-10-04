import functools
from typing import List, Set, Union, Generic, Callable
from typing import Iterator, Dict, Type, TypeVar, cast, Optional
from enum import Enum
from dataclasses import dataclass
import z3  # type: ignore


WeightFunc = Callable[[Type["TestOption"]], Optional[float]]


class UnsolvableError(Exception):
    pass


class Constraint:
    def _to_z3_formula(self):
        raise NotImplementedError("This is an abstract base class/group class")

    def __iter__(self):
        raise NotImplementedError("This is an abstract base class/group class")

    def __and__(self, other: "Constraint"):
        return And(self, other)

    def __or__(self, other: "Constraint"):
        return Or(self, other)

    def __xor__(self, other: "Constraint"):
        return Xor(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class And(Constraint):
    op1: Constraint
    op2: Constraint

    def _to_z3_formula(self):
        return z3.And(self.op1._to_z3_formula(), self.op2._to_z3_formula())

    def __iter__(self):
        yield from self.op1
        yield from self.op2


@dataclass(frozen=True)
class And(Constraint):
    op1: Constraint
    op2: Constraint

    def _to_z3_formula(self):
        return z3.And(self.op1._to_z3_formula(), self.op2._to_z3_formula())

    def __iter__(self):
        yield from self.op1
        yield from self.op2


@dataclass(frozen=True)
class Not(Constraint):
    op: Constraint

    def _to_z3_formula(self):
        return z3.Not(self.op._to_z3_formula())

    def __iter__(self):
        yield from self.op


@dataclass(frozen=True)
class Or(Constraint):
    op1: Constraint
    op2: Constraint

    def _to_z3_formula(self):
        return z3.Or(self.op1._to_z3_formula(), self.op2._to_z3_formula())

    def __iter__(self):
        yield from self.op1
        yield from self.op2


@dataclass(frozen=True)
class Xor(Constraint):
    op1: Constraint
    op2: Constraint

    def _to_z3_formula(self):
        return z3.Xor(self.op1._to_z3_formula(), self.op2._to_z3_formula())

    def __iter__(self):
        yield from self.op1
        yield from self.op2


@dataclass(frozen=True)
class Implies(Constraint):
    op1: Constraint
    op2: Constraint

    def _to_z3_formula(self):
        return z3.Implies(self.op1._to_z3_formula(), self.op2._to_z3_formula())

    def __iter__(self):
        yield from self.op1
        yield from self.op2


@dataclass(frozen=True)
class Equal(Constraint):
    op1: Constraint
    op2: Constraint

    def _to_z3_formula(self):
        return self.op1._to_z3_formula() == self.op2._to_z3_formula()

    def __iter__(self):
        yield from self.op1
        yield from self.op2


@dataclass(frozen=True)
class SetOperator(Constraint):
    op: Type["TestOption"]
    where: Optional[Callable[[Type["TestOption"]], bool]] = None
    excluding: Optional[Set[Type["TestOption"]]] = None

    def __iter__(self):
        impls = self.op.implementations
        if self.where:
            impls = {i for i in impls if self.where(i)}
        if self.excluding:
            impls = impls - self.excluding
        return iter(impls)


@dataclass(frozen=True)
class No(SetOperator):
    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.And(*[z3.Not(impl._to_z3_formula()) for impl in self])


@dataclass(frozen=True)
class Any(SetOperator):
    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.Or(*[impl._to_z3_formula() for impl in self])


@dataclass(frozen=True)
class All(SetOperator):
    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.And(*[c._to_z3_formula() for c in self])


def _count_ones(z3_vars, start, end):
    if end - start == 1:
        return z3_vars[start], False
    else:
        exact_one1, more_than_one1 = _count_ones(z3_vars, start, (start + end) // 2)
        exact_one2, more_than_one2 = _count_ones(z3_vars, (start + end) // 2, end)
        more_than_one = z3.Or(
            z3.And(exact_one1, exact_one2), more_than_one1, more_than_one2
        )
        exact_one = z3.And(z3.Xor(exact_one1, exact_one2), z3.Not(more_than_one))
        return exact_one, more_than_one


@dataclass(frozen=True)
class MaxOne(SetOperator):
    @functools.lru_cache
    def _to_z3_formula(self):
        z3_vars = [impl._to_z3_formula() for impl in self.op.implementations]
        if len(z3_vars) == 0:
            return True
        else:
            _, more_than_one = _count_ones(z3_vars, 0, len(z3_vars))
            return z3.Not(more_than_one)


@dataclass(frozen=True)
class ExactOne(SetOperator):
    @functools.lru_cache
    def _to_z3_formula(self):
        z3_vars = [impl._to_z3_formula() for impl in self.op.implementations]
        if len(z3_vars) == 0:
            return False
        else:
            exact_one, _ = _count_ones(z3_vars, 0, len(z3_vars))
            return exact_one


def requires(constraint: Constraint, *, by: Optional[Type["TestOption"]] = None):
    if by is None:

        def decorator(cls: Type[TestOption]):
            cls.constraints.append(Implies(cls, constraint))
            return cls

        return decorator
    else:
        by.constraints.append(Implies(by, constraint))


def link(constraint: Constraint, testoption: Optional[Type["TestOption"]] = None):
    if testoption is None:

        def decorator(cls: Type[TestOption]):
            cls.constraints.append(Equal(cls, constraint))
            return cls

        return decorator
    else:
        testoption.constraints.append(Equal(testoption, constraint))


class TestOptionMeta(type, Constraint):
    def __repr__(self):
        return "<" + self.__name__ + ">"

    def __iter__(self):
        yield self


T = TypeVar("T", bound="TestOption")


class CountSpec(Enum):
    ZERO_OR_ONE = 0
    ANY = 1
    EXACT_ONE = 2


@dataclass
class BindInfo:
    type: Type["TestOption"]
    count_spec: CountSpec


def bind(type: Type[T]) -> T:
    return cast(T, BindInfo(type, CountSpec.EXACT_ONE))


def bind_optional(type: Type[T]) -> Optional[T]:
    return cast(Optional[T], BindInfo(type, CountSpec.ZERO_OR_ONE))


def bind_set(type: Type[T]) -> Set[T]:
    return cast(Set[T], BindInfo(type, CountSpec.ANY))


def avail_impls(bound_attr: Union[T, Set[T], None]) -> Set[T]:
    """
    Returns all available subclasses that are available for the passed
    attribute.

    bound_attr has to be a class attribute that was setup via bind(),
    bind_optional() or bind_set().
    """
    if isinstance(bound_attr, (type(None), set, TestOption)):
        raise ValueError(
            "implementations() works only at class level, "
            "not on attributes of instiated classes"
        )
    bind_info = cast(BindInfo, bound_attr)
    return cast(Set[T], bind_info.type.implementations)


class TestHandler:
    pass


TH = TypeVar("TH", bound=TestHandler)


def handler(type: Type[TH]) -> Optional[TH]:
    if not issubclass(type, TestHandler):
        raise TypeError(f"'Expected subtype of 'TestHandler', got {type!r}")
    return cast(Optional[TH], type)


@dataclass
class TestSetup(Generic[T]):
    root: Type[T]
    testoptions: Dict[Type["TestOption"], "TestOption"]

    @property
    def root_inst(self) -> T:
        return cast(T, self.testoptions[self.root])

    def run(self, *testhandlers: "TestHandler") -> T:
        handler_map = {type(h): h for h in testhandlers}
        if len(testhandlers) != len(handler_map):
            raise ValueError(
                "All TestHandlers require unique type for correct identification"
            )
        for impl in self.root.iter_dependencies():
            if impl in self.testoptions:
                to = self.testoptions[impl]
                for name, handler_cls in to.handlers.items():
                    setattr(to, name, handler_map[handler_cls])
                to.setup()
        self.root_inst.test()
        for impl in reversed(list(self.root.iter_dependencies())):
            if impl in self.testoptions:
                self.testoptions[impl].teardown()
        return self.root_inst


class TestOption(metaclass=TestOptionMeta):
    __z3var: Optional[z3.Bool] = None
    group = True
    weight: Optional[float] = None
    constraints: List[Constraint] = []
    implementations: Set[Type["TestOption"]] = set()
    bindings: Dict[str, BindInfo] = dict()
    handlers: Dict[str, Type[TestHandler]] = dict()

    def __init_subclass__(
        cls, *, group: Optional[bool] = False, weight: Optional[float] = None, **kwargs
    ):
        super().__init_subclass__(**kwargs)
        cls.implementations = set()
        cls.constraints = cls.constraints[:]  # create copy
        cls.bindings = cls.bindings.copy()
        cls.handlers = cls.handlers.copy()
        for nm, val in cls.__dict__.copy().items():
            if isinstance(val, BindInfo):
                if not issubclass(val.type, TestOption):
                    raise TypeError(
                        f"'{cls.__name__}.{nm}' is of type '{val.type!r}' "
                        f"(has to be a subclass of 'TestOption')"
                    )
                cls.bindings[nm] = val
            elif isinstance(val, type) and issubclass(val, TestHandler):
                cls.handlers[nm] = val
                setattr(cls, nm, None)
        if group and weight:
            raise ValueError("TestOption group can have no weigth")
        if not group:
            cls.__z3var = z3.Bool(f"{cls.__name__}@{id(cls)}")
        cls.weight = weight
        cls.group = group
        for attrname, binding in cls.bindings.items():
            if binding.count_spec == CountSpec.EXACT_ONE:
                cls.constraints.append(Implies(cls, ExactOne(binding.type)))
            elif binding.count_spec == CountSpec.ZERO_OR_ONE:
                cls.constraints.append(
                    Implies(cls, No(binding.type) | ExactOne(binding.type))
                )
        if not group:
            cls.__register_at_parent_classes()

    def __init__(self):
        if self.group:
            raise NotImplementedError(
                "This is a TestOption group which must not be instantiated"
            )

    @classmethod
    def __register_at_parent_classes(cls):
        for basecls in cls.__mro__:
            if issubclass(basecls, TestOption) and basecls is not TestOption:
                assert basecls is cls or basecls.group, (
                    f"Baseclass {basecls!r} of {cls!r} "
                    f"has to be marked as 'class ...(..., group=True):'"
                )
                basecls.register_implementation(cls)

    @classmethod
    def register_implementation(cls, impl_cls: Type["TestOption"]):
        cls.implementations.add(impl_cls)

    @classmethod
    def _to_z3_formula(cls):
        if cls.__z3var is not None:
            return cls.__z3var
        else:
            raise ValueError("TestOption groups cannot be converted to Z3 formulars")

    @classmethod
    def iter_dependencies(
        cls, collected_deps: Set[Type["TestOption"]] = None
    ) -> Iterator[Type["TestOption"]]:
        collected_deps = collected_deps if collected_deps is not None else set()
        for impl in cls.implementations:
            if impl in collected_deps:
                continue
            collected_deps.add(impl)
            for binding in impl.bindings.values():
                if binding.type not in collected_deps:
                    yield from binding.type.iter_dependencies(collected_deps)
            for constr in impl.constraints:
                for constr_depcls in constr:
                    if constr_depcls not in collected_deps:
                        yield from constr_depcls.iter_dependencies(collected_deps)
            yield impl

    T = TypeVar("T", bound="TestOption")

    @classmethod
    def get_absolute_weights(
        cls: Type[T], *constrs: Constraint, weight_func: Optional[WeightFunc] = None
    ):
        def collect_parent_child_relations(child: Type["TestOption"]):
            parent = cast(Type["TestOption"], child.__base__)
            if parent is TestOption:
                root_list.add(child)
            elif parent in children_map:
                children_map[parent].add(child)
            else:  # not yet visited => step up to parent
                children_map[parent] = {child}
                collect_parent_child_relations(parent)

        def sum_abs_weights(parent: Type["TestOption"]):
            weight = weight_func(parent)
            if parent.group:
                abs_weights[parent] = sum(
                    sum_abs_weights(child) for child in children_map[parent]
                )
            else:
                abs_weights[parent] = weight or 1
            return weight if weight is not None else abs_weights[parent]

        def scale_abs_weights(parent: Type["TestOption"], factor=1.0):
            dest_parent_weight = weight_func(parent)
            if dest_parent_weight is None or abs_weights[parent] == dest_parent_weight:
                child_factor = factor  # no rescaling required
            elif abs_weights[parent] == 0:
                raise ValueError(
                    f"{parent!r} enforces weigth {dest_parent_weight} "
                    f"but has no children (or only children of weight 0)"
                )
            else:
                child_factor = factor * dest_parent_weight / abs_weights[parent]
            abs_weights[parent] *= child_factor
            for child in children_map[parent]:
                scale_abs_weights(child, child_factor)

        if weight_func is None:
            weight_func = lambda impl: impl.weight

        available_testoptions = set(cls.iter_dependencies()).union(*constrs)
        children_map: Dict[Type["TestOption"], Set[Type["TestOption"]]]
        children_map = dict.fromkeys(available_testoptions, set())
        root_list = set()
        for to in available_testoptions:
            collect_parent_child_relations(to)

        abs_weights = {}
        for root in root_list:
            sum_abs_weights(root)
            scale_abs_weights(root)
        return abs_weights

    @classmethod
    def generate_testsetup(
        cls: Type[T],
        *constrs: Constraint,
        weight_func: Optional[WeightFunc] = None,
    ) -> TestSetup[T]:
        def instanciate_matching_testoptions():
            solver = z3.Optimize()
            solver.add(cls.__z3var)
            for constr in constrs:
                solver.add(constr._to_z3_formula())
            available_testoptions = set(cls.iter_dependencies()).union(*constrs)
            for to in available_testoptions:
                for constr in to.constraints:
                    solver.add(constr._to_z3_formula())
            abs_weights = cls.get_absolute_weights(*constrs, weight_func=weight_func)
            for to, abs_weight in abs_weights.items():
                if not to.group:
                    if abs_weight < 0.000001:
                        raise ValueError("Weights < 0.000001 are not supported")
                    solver.add_soft(to._to_z3_formula(), abs_weight)
            if solver.check() != z3.sat:
                raise UnsolvableError("Cannot solve constraints")
            model = solver.model()
            return {to: to() for to in available_testoptions if model[to.__z3var]}

        def bind_instances(testoptions: Dict[Type[TestOption], TestOption]):
            for to in testoptions.values():
                for attrname, binding in to.bindings.items():
                    bound_testoptions = {
                        testoptions[impl]
                        for impl in binding.type.implementations
                        if impl in testoptions
                    }
                    if binding.count_spec == CountSpec.ANY:
                        setattr(to, attrname, bound_testoptions)
                    else:
                        impl = bound_testoptions.pop() if bound_testoptions else None
                        setattr(to, attrname, impl)

        if cls.group:
            raise UnsolvableError("Cannot generate TestSetup from TestOption group")
        testoptions = instanciate_matching_testoptions()
        bind_instances(testoptions)
        return TestSetup(cls, testoptions)

    def setup(self):
        pass

    def test(self):
        pass

    def teardown(self):
        pass
