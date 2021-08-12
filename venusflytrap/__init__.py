import functools
from typing import List, Set, Union, Generic, FrozenSet
from typing import Iterator, Dict, Type, TypeVar, cast, Optional
from enum import Enum
from dataclasses import dataclass
import z3  # type: ignore


class UnsolvableError(Exception):
    pass


class Constraint:
    def _to_z3_formula(self):
        raise NotImplementedError("This is an abstract base class")

    def __iter__(self):
        raise NotImplementedError("This is an abstract base class")

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
class No(Constraint):
    op: Type["TestOption"]

    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.And(
            *[z3.Not(impl._to_z3_formula()) for impl in self.op.implementations]
        )

    def __iter__(self):
        yield self.op


@dataclass(frozen=True)
class Any(Constraint):
    op: Type["TestOption"]

    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.Or(*[impl._to_z3_formula() for impl in self.op.implementations])

    def __iter__(self):
        yield self.op


@dataclass(frozen=True)
class ExactOne(Constraint):
    op: Type["TestOption"]

    @functools.lru_cache
    def _to_z3_formula(self):
        def count_ones(start, end):
            if end - start == 1:
                return z3_vars[start], False
            else:
                exact_one1, more_than_one1 = count_ones(start, (start + end) // 2)
                exact_one2, more_than_one2 = count_ones((start + end) // 2, end)
                more_than_one = z3.Or(
                    z3.And(exact_one1, exact_one2), more_than_one1, more_than_one2
                )
                exact_one = z3.And(
                    z3.Xor(exact_one1, exact_one2), z3.Not(more_than_one)
                )
                return exact_one, more_than_one

        z3_vars = [impl._to_z3_formula() for impl in self.op.implementations]
        if len(z3_vars) == 0:
            return False
        else:
            exact_one, _ = count_ones(0, len(z3_vars))
            return exact_one

    def __iter__(self):
        yield self.op


@dataclass(frozen=True)
class DisableAllExcept(Constraint):
    exc_cls_list: FrozenSet[Type["TestOption"]]
    base_cls: Type["TestOption"]

    def __init__(self, *exc_cls_list, of=None):
        # this is a workaround required for __init__ in frozen dataclasses:
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if of is None:
            ValueError("DisableAllExcept() has to be called with 'of' parameter")
        object.__setattr__(self, "exc_cls_list", frozenset(exc_cls_list))
        object.__setattr__(self, "base_cls", of)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{', '.join(c.__name__ for c in self.exc_cls_list)}, "
            f"of={self.base_cls.__name__})"
        )

    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.And(*[z3.Not(c._to_z3_formula()) for c in self])

    def __iter__(self):
        return iter(self.base_cls.implementations - self.exc_cls_list)


def requires(constraint: Constraint):
    def decorator(cls: Type[TestOption]):
        cls.constraints.append(Implies(cls, constraint))
        return cls

    return decorator


class TestOptionMeta(type, Constraint):
    def __repr__(self):
        return "<" + self.__name__ + ">"

    def __iter__(self):
        yield self


T = TypeVar("T", bound="TestOption")


@dataclass
class BindInfo:
    class CountSpec(Enum):
        ZERO_OR_ONE = 0
        ANY = 1
        EXACT_ONE = 2

    type: "TestOption"
    count_spec: CountSpec = CountSpec.EXACT_ONE


def bind(type: Type[T]) -> Union[T, Type[T]]:
    return cast(Type[T], BindInfo(type))


def bind_optional(type: Type[T]) -> Union[Optional[T], Type[T]]:
    result = BindInfo(type, BindInfo.CountSpec.ZERO_OR_ONE)
    return cast(Type[T], result)


def bind_set(type: Type[T]) -> Union[Set[T], Type[T]]:
    result = BindInfo(type, BindInfo.CountSpec.ANY)
    return cast(Type[T], result)


class TestSetup(Generic[T]):
    def __init__(
        self, root: Type[T], testoptions: Dict[Type["TestOption"], "TestOption"]
    ):
        self.root = root
        self.testoptions = testoptions

    @property
    def root_inst(self) -> T:
        return cast(T, self.testoptions[self.root])

    def run(self) -> T:
        for impl in self.root.iter_dependencies():
            if impl in self.testoptions:
                self.testoptions[impl].setup()
        self.root_inst.test()
        for impl in reversed(list(self.root.iter_dependencies())):
            if impl in self.testoptions:
                self.testoptions[impl].teardown()
        return self.root_inst


class TestOption(metaclass=TestOptionMeta):
    __z3var: z3.Bool
    __abstract = True
    constraints: List[Constraint] = []
    implementations: Set[Type["TestOption"]] = set()
    bindings: Dict[str, BindInfo] = dict()

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.implementations = set()
        cls.constraints = cls.constraints[:]  # create copy
        cls.bindings = cls.bindings.copy()
        for nm, val in cls.__dict__.copy().items():
            if isinstance(val, BindInfo):
                cls.bindings[nm] = val
                setattr(cls, nm, val.type)
        cls.__z3var = z3.Bool(f"{cls.__name__}@{id(cls)}")
        cls.__abstract = abstract
        for attrname, binding in cls.bindings.items():
            if not issubclass(binding.type, TestOption):
                raise TypeError(
                    f"'{cls.__name__}.{attrname}' is of type '{binding.type!r}' "
                    f"(has to be a subclass of 'TestOption')"
                )
            elif binding.count_spec == BindInfo.CountSpec.EXACT_ONE:
                cls.constraints.append(Implies(cls, ExactOne(binding.type)))
            elif binding.count_spec == BindInfo.CountSpec.ZERO_OR_ONE:
                cls.constraints.append(
                    Implies(cls, No(binding.type) | ExactOne(binding.type))
                )
        if not abstract:
            cls.__register_at_parent_classes()

    def __init__(self):
        if self.__abstract:
            raise NotImplementedError(
                "This is an abstract TestOption which must not be instantiated"
            )

    @classmethod
    def __register_at_parent_classes(cls):
        for basecls in cls.__mro__:
            if issubclass(basecls, TestOption) and basecls is not TestOption:
                assert basecls is cls or basecls.__abstract, (
                    f"Baseclass {basecls!r} of {cls!r} "
                    f"has to be marked as 'class ...(..., abstract=True):'"
                )
                basecls.register_implementation(cls)

    @classmethod
    def register_implementation(cls, impl_cls: Type["TestOption"]):
        cls.implementations.add(impl_cls)

    @classmethod
    def _to_z3_formula(cls):
        return cls.__z3var

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
    def generate_testsetup(cls: Type[T], *constrs: Constraint) -> TestSetup[T]:
        def instanciate_matching_testoptions():
            available_testoptions = set(cls.iter_dependencies())
            solver = z3.Solver()
            solver.add(cls._to_z3_formula())
            for constr in constrs:
                solver.add(constr._to_z3_formula())
                available_testoptions |= set(constr)
            for to in available_testoptions:
                for constr in to.constraints:
                    solver.add(constr._to_z3_formula())
            if solver.check() != z3.sat:
                raise UnsolvableError("Cannot solve constraints")
            model = solver.model()
            return {
                to: to() for to in available_testoptions if model[to._to_z3_formula()]
            }

        def bind_instances(testoptions: Dict[Type[TestOption], TestOption]):
            for to in testoptions.values():
                for attrname, binding in to.bindings.items():
                    bound_testoptions = {
                        testoptions[impl]
                        for impl in binding.type.implementations
                        if impl in testoptions
                    }
                    if binding.count_spec == BindInfo.CountSpec.ANY:
                        setattr(to, attrname, bound_testoptions)
                    else:
                        impl = bound_testoptions.pop() if bound_testoptions else None
                        setattr(to, attrname, impl)

        testoptions = instanciate_matching_testoptions()
        bind_instances(testoptions)
        return TestSetup(cls, testoptions)

    def setup(self):
        pass

    def test(self):
        pass

    def teardown(self):
        pass
