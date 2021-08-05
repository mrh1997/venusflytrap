from __future__ import annotations
import functools
from typing import (
    get_origin,
    get_args,
    get_type_hints,
    List,
    Set,
    Union,
    Type,
    TypeVar,
    Tuple,
    Iterator,
    Dict,
)
from dataclasses import dataclass
import z3


class UnsolvableError(Exception):
    pass


class Constraint:
    def _to_z3_formula(self):
        raise NotImplementedError("This is an abstract base class")

    def __iter__(self):
        raise NotImplementedError("This is an abstract base class")

    def __and__(self, other: Constraint):
        return And(self, other)

    def __or__(self, other: Constraint):
        return Or(self, other)

    def __xor__(self, other: Constraint):
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
    op: Type[TestOption]

    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.And(
            *[z3.Not(impl._to_z3_formula()) for impl in self.op.implementations]
        )

    def __iter__(self):
        yield self.op


@dataclass(frozen=True)
class Any(Constraint):
    op: Type[TestOption]

    @functools.lru_cache
    def _to_z3_formula(self):
        return z3.Or(*[impl._to_z3_formula() for impl in self.op.implementations])

    def __iter__(self):
        yield self.op


@dataclass(frozen=True)
class ExactOne(Constraint):
    op: Type[TestOption]

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


def requires(constraint: Constraint):
    def decorator(cls: Type[TestOption]):
        cls.constraints.append(Implies(cls, constraint))
        return cls

    return decorator


class TestOptionMeta(type, Constraint):
    pass

    def __repr__(self):
        return "<" + self.__name__ + ">"

    def __iter__(self):
        yield self


class TestOption(metaclass=TestOptionMeta):
    __z3var: z3.Bool
    __abstract = True
    constraints: List[Constraint] = []
    implementations: Set[Type[TestOption]] = set()

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.implementations = set()
        cls.constraints = cls.constraints[:]  # create copy
        cls.__z3var = z3.Bool(f"{cls.__name__}@{id(cls)}")
        cls.__abstract = abstract
        for attrname, depcls, count_spec in cls._direct_dependencies():
            if not issubclass(depcls, TestOption):
                raise TypeError(
                    f"'{cls.__name__}.{attrname}' is of type '{depcls!r}' "
                    f"(has to be a subclass of 'TestOption')"
                )
            elif count_spec == "EXACT_ONE":
                cls.constraints.append(Implies(cls, ExactOne(depcls)))
            elif count_spec == "ZERO_OR_ONE":
                cls.constraints.append(Implies(cls, No(depcls) | ExactOne(depcls)))
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
    def register_implementation(cls, impl_cls: Type[TestOption]):
        cls.implementations.add(impl_cls)

    @classmethod
    def _to_z3_formula(cls):
        return cls.__z3var

    @classmethod
    def iter_dependencies(
        cls, collected_deps: Set[Type[TestOption]] = None
    ) -> Iterator[Type[TestOption]]:
        collected_deps = collected_deps if collected_deps is not None else set()
        for impl in cls.implementations:
            if impl in collected_deps:
                continue
            collected_deps.add(impl)
            for _, direct_depcls, _ in impl._direct_dependencies():
                if direct_depcls not in collected_deps:
                    yield from direct_depcls.iter_dependencies(collected_deps)
            for constr in impl.constraints:
                for constr_depcls in constr:
                    if constr_depcls not in collected_deps:
                        yield from constr_depcls.iter_dependencies(collected_deps)
            yield impl

    @classmethod
    @functools.lru_cache()
    def _direct_dependencies(cls) -> Set[Tuple[str, Type[TestOption], str]]:
        result = set()
        initialized_attrs = dir(cls)
        for attrname, depcls in get_type_hints(cls).items():
            if attrname in initialized_attrs:
                continue
            if get_origin(depcls) is Union and get_args(depcls)[1:2] == (type(None),):
                depcls = get_args(depcls)[0]
                count_spec = "ZERO_OR_ONE"
            elif get_origin(depcls) is set:
                depcls = get_args(depcls)[0]
                count_spec = "ANY"
            else:
                count_spec = "EXACT_ONE"
            result.add((attrname, depcls, count_spec))
        return result


T = TypeVar("T", bound=TestOption)


def generate_testsetup(root_testoption: Type[T], *constrs: Constraint) -> T:
    def instanciate_matching_testoptions():
        available_testoptions = set(root_testoption.iter_dependencies())
        solver = z3.Solver()
        solver.add(root_testoption._to_z3_formula())
        for constr in constrs:
            solver.add(constr._to_z3_formula())
            available_testoptions |= set(constr)
        for to in available_testoptions:
            for constr in to.constraints:
                solver.add(constr._to_z3_formula())
        if solver.check() != z3.sat:
            raise UnsolvableError("Cannot solve constraints")
        model = solver.model()
        return {to: to() for to in available_testoptions if model[to._to_z3_formula()]}

    def link_instances(testoptions: Dict[Type[TestOption], TestOption]):
        for to in testoptions.values():
            for attrname, cls, count_spec in to._direct_dependencies():
                impls = {
                    testoptions[impl]
                    for impl in cls.implementations
                    if impl in testoptions
                }
                if count_spec == "ANY":
                    setattr(to, attrname, impls)
                else:
                    impl = impls.pop() if impls else None
                    setattr(to, attrname, impl)

    testoptions = instanciate_matching_testoptions()
    link_instances(testoptions)
    return testoptions[root_testoption]
