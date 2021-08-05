import pytest
from venusflytrap import (
    TestOption,
    Constraint,
    ExactOne,
    No,
    Any,
    Implies,
    Type,
    requires,
    generate_testsetup,
    And,
    UnsolvableError,
)
from typing import Optional, Set
import z3


class TestTestOption:
    def test_create_onAbstractWithoutImplementations_setImplementationsEmpty(self):
        class A(TestOption, abstract=True):
            pass

        assert A.implementations == set()

    def test_create_onImplementation_setImplementationsToSelf(self):
        class Impl(TestOption):
            pass

        assert Impl.implementations == {Impl}

    def test_create_onMultipleAbstractBases_addsSelfToBasesImplementationLists(self):
        class A(TestOption, abstract=True):
            pass

        class B(TestOption, abstract=True):
            pass

        class Impl(A, B):
            pass

        assert all(cls.implementations == {Impl} for cls in [A, B])

    def test_create_onMulipleImplementations_addsAllToBaseClassImplementations(self):
        class A(TestOption, abstract=True):
            pass

        class Impl1(A):
            pass

        class Impl2(A):
            pass

        assert A.implementations == {Impl1, Impl2}

    def test_create_onParentHasMoreImplsThanChild_maintainsDifferentImplLists(self):
        class A(TestOption, abstract=True):
            pass

        class B(A, abstract=True):
            pass

        class ImplA(A):
            pass

        class ImplB(B):
            pass

        assert A.implementations == {ImplA, ImplB}
        assert B.implementations == {ImplB}

    def test_create_onImplementation_callsRegisterImplementionOnAllParentClasses(self):
        class A(TestOption, abstract=True):
            registered_children = set()

            @classmethod
            def register_implementation(cls, impl_cls: Type[TestOption]):
                super().register_implementation(impl_cls)
                cls.registered_children.add(impl_cls)

        class B(A, abstract=True):
            pass

        class Impl1(B):
            pass

        class Impl2(B):
            pass

        assert A.registered_children == {Impl1, Impl2}

    def test_instanciate_onAbstractType_raisesNotImplementedError(self):
        class A(TestOption, abstract=True):
            pass

        with pytest.raises(NotImplementedError):
            A()

    def test_iterDependencies_onImplType_returnsOnlySelf(self):
        class Impl(TestOption):
            pass

        assert set(Impl.iter_dependencies()) == {Impl}

    def test_iterDependencies_onAbstractType_returnsEmpty(self):
        class A(TestOption, abstract=True):
            pass

        assert set(A.iter_dependencies()) == set()

    def test_create_onAttrWithoutImpls_fails(self):
        class A(TestOption, abstract=True):
            pass

        class B(TestOption, abstract=True):
            pass

        class C(TestOption, abstract=True):
            pass

        class Impl(TestOption):
            dep: A
            opt_dep: Optional[B]
            set_dep: Set[C]

        assert set(Impl.constraints) == {
            Implies(Impl, ExactOne(A)),
            Implies(Impl, No(B) | ExactOne(B)),
        }

    def test_create_onWrongAttrType_raiseMeaningfulTypeError(self):
        with pytest.raises(TypeError) as exc:

            class Impl(TestOption):
                dep: int

        assert "Impl.dep" in str(exc)

    def test_requiresDecorator_onAbstractClass_addsConstraintsToImpls(self):
        a_constr, b_constr = Constraint(), Constraint()

        @requires(a_constr)
        class A(TestOption, abstract=True):
            pass

        @requires(b_constr)
        class B(A):
            pass

        assert set(A.constraints) == {Implies(A, a_constr)}
        assert set(B.constraints) == {Implies(A, a_constr), Implies(B, b_constr)}

    def test_requiresDecorator_onImpl_doesNotModifyConstraintsOfSibling(self):
        class A(TestOption, abstract=True):
            pass

        @requires(Constraint())
        class Impl(A):
            pass

        class SiblingImpl(A):
            pass

        assert len(SiblingImpl.constraints) == 0

    def test_iterDependencies_onRecursiveDepenedencies_ok(self):
        class A(TestOption, abstract=True):
            pass

        class B(TestOption, abstract=True):
            pass

        class ImplA(A):
            pass

        class ImplB(B):
            dep: A

        class ImplC(TestOption):
            dep: B

        assert set(ImplC.iter_dependencies()) == {ImplA, ImplB, ImplC}

    def test_iterDependencies_onOptionalAndSetAttrs_ok(self):
        class A(TestOption, abstract=True):
            pass

        class ImplA(A):
            pass

        class B(TestOption, abstract=True):
            pass

        class ImplB(B):
            pass

        class C(TestOption):
            opt_dep: Optional[A]
            set_dep: Set[B]

        assert set(C.iter_dependencies()) == {C, ImplA, ImplB}

    def test_iterDependencies_onReferredByConstraintsOnly_addedToList(self):
        class Impl1(TestOption):
            pass

        class Impl2(TestOption):
            dep: Impl1

        @requires(Impl2)
        class Impl3(TestOption):
            pass

        assert set(Impl3.iter_dependencies()) == {Impl1, Impl2, Impl3}

    def test_iterDependencies_orderedFromLeafToRoot(self):
        class ImplLeaf(TestOption):
            pass

        class ImplMiddle1(TestOption):
            dep: ImplLeaf

        class ImplMiddle2(TestOption):
            dep: ImplLeaf

        class ImplRoot(TestOption):
            dep1: ImplMiddle2
            dep2: ImplMiddle1

        leaf, *middle, root = list(ImplRoot.iter_dependencies())
        assert leaf == ImplLeaf
        assert set(middle) == {ImplMiddle1, ImplMiddle2}
        assert root == ImplRoot


class TestConstraint:
    def assert_solution(self, solution, expr_under_test):
        class A(TestOption, abstract=True):
            pass

        impls = []
        for c in range(expr_under_test.__code__.co_argcount - 1):

            class Impl(A):
                pass

            impls.append(Impl)

        s = z3.Solver()
        s.add(*[i._to_z3_formula() for i in expr_under_test(A, *impls)])
        if s.check() == z3.unsat:
            raise ValueError()
        else:
            m = s.model()
            for ndx, impl in enumerate(impls, 1):
                exp_val = ndx in solution
                err_text = f"'I{ndx}' is {not exp_val} instead of {exp_val}"
                assert m[impl._to_z3_formula()] == exp_val, err_text

    def assert_nosolution(self, expr_under_test):
        try:
            self.assert_solution({}, expr_under_test)
        except ValueError:
            pass
        else:
            raise AssertionError("expected unsatisfiable solution")

    def test_create_createsBoolVarWithUniqueName(self):
        class Impl(TestOption):
            pass

        OldImpl = Impl

        class Impl(TestOption):
            pass

        s = z3.Solver()
        s.add(z3.Xor(OldImpl._to_z3_formula(), Impl._to_z3_formula()))
        assert s.check() == z3.sat

    def test_operators_onImplementations_ok(self):
        self.assert_solution({1, 2}, lambda A, I1, I2: [I1 & I2])
        self.assert_solution({2}, lambda A, I1, I2: [~I1, I1 | I2])

    def test_operators_onImplies_ok(self):
        self.assert_solution({1, 2}, lambda A, I1, I2: [I1, Implies(I1, I2)])
        self.assert_solution({2}, lambda A, I1, I2: [~I1, I2, Implies(I1, I2)])
        self.assert_solution({}, lambda A, I1, I2: [~I1, ~I2, Implies(I1, I2)])

    def test_No_onAbstractClassWithUndecidedImpls_ok(self):
        self.assert_solution({}, lambda A, I1, I2, I3: [No(A)])

    def test_No_onAbstractClassWithASetImpls_unsatisfiable(self):
        self.assert_nosolution(lambda A, I1, I2, I3: [I2, No(A)])

    def test_Any_onAbstractClassWithAllImplsDisabled_unsatisfiable(self):
        self.assert_nosolution(lambda A, I1: [~I1, Any(A)])

    def test_Any_onAbstractClassWithAllImplsDisabled_Ok(self):
        self.assert_nosolution(lambda A, I1, I2: [~I1, ~I2, Any(A)])

    def test_ExactOne_onAbstractClassWithoutImpls_unsatisfiable(self):
        self.assert_nosolution(lambda A: [ExactOne(A)])

    def test_ExactOne_onAbstractClassWithUndecidedImpls_ok(self):
        self.assert_solution({2}, lambda A, I1, I2, I3: [I2, ExactOne(A)])

    def test_ExactOne_onAbstractClassWithTwoAlreadySetImpls_unsatisfiable(self):
        self.assert_nosolution(lambda A, I1, I2: [~I1, ~I2, ExactOne(A)])

    def test_ExactOne_onMultipleCalls_optimizeByReturnsSameResult(self):
        class A(TestOption, abstract=True):
            pass

        class Impl1(A):
            pass

        class Impl2(A):
            pass

        assert ExactOne(A)._to_z3_formula() is ExactOne(A)._to_z3_formula()
        assert Any(A)._to_z3_formula() is Any(A)._to_z3_formula()
        assert No(A)._to_z3_formula() is No(A)._to_z3_formula()

    def test_iter_onTestOption_returnsSelf(self):
        class A(TestOption, abstract=True):
            pass

        assert set(iter(A)) == {A}

    def test_iter_onLogicalOps_returnsOps(self):
        class Impl1(TestOption):
            pass

        class Impl2(TestOption):
            pass

        assert set(iter(Impl1 & ~Impl2)) == {Impl1, Impl2}

    def test_iter_onAbstractClassOps_returnsAbstractCls(self):
        class A(TestOption):
            pass

        assert set(iter(ExactOne(A))) == {A}
        assert set(iter(No(A))) == {A}
        assert set(iter(Any(A))) == {A}


class TestGenerateTestSetup:
    def create_impls(self, cnt: int, Base: Type[TestOption] = TestOption):
        for c in range(cnt):

            class Impl(Base):
                pass

            yield Impl

    def test_generateTestsetup_instanciatesTestOption(self):
        class Impl(TestOption):
            def __init__(self):
                super().__init__()
                self.val = 123

        ts = generate_testsetup(Impl)
        assert ts.val == 123

    def test_generateTestsetup_instanciatesAndLinksDependendImpls(self):
        class A(TestOption, abstract=True):
            val = 1

        class ImplA(A):
            def __init__(self):
                super().__init__()
                self.val = 123

        class Impl(TestOption):
            dep: A

        ts = generate_testsetup(Impl)
        assert ts.dep.val == 123

    def test_generateTestsetup_onAdditionalConstraints_instantiatesImpls(self):
        class A(TestOption, abstract=True):
            def __init__(self):
                nonlocal inst_count
                super().__init__()
                inst_count += 1

        implAs = list(self.create_impls(10, A))

        inst_count = 0
        generate_testsetup(implAs[0], implAs[1] & implAs[2], implAs[4])
        assert inst_count == 4

    def test_generateTestsetup_onOptionalDepAndNoImpl_setsDepToNone(self):
        class A(TestOption, abstract=True):
            pass

        class Impl(TestOption):
            dep: Optional[A]

        ts = generate_testsetup(Impl)
        assert ts.dep is None

    @pytest.mark.parametrize("impl_count", [0, 1, 2])
    def test_generateTestsetup_onSetDep_returnsSet(self, impl_count):
        class A(TestOption, abstract=True):
            pass

        implAs = {type("ImplA", (A,), {}) for c in range(impl_count)}

        class Impl(TestOption):
            dep: Set[A]

        ts = generate_testsetup(Impl, *implAs)
        assert {type(to) for to in ts.dep} == implAs

    def test_generateTestsetup_onSelectExactOneImpl_ok(self):
        class A(TestOption, abstract=True):
            pass

        def __init__(self):
            nonlocal inst_cnt
            inst_cnt += 1

        implAs = [type("ImplA", (A,), {"__init__": __init__}) for c in range(10)]

        class Impl(TestOption):
            dep: A

        inst_cnt = 0
        ts = generate_testsetup(Impl)
        assert isinstance(ts.dep, tuple(implAs))
        assert inst_cnt == 1

    def test_generateInstance_onUnsolvable_raisesUnsolvableError(self):
        class Impl(TestOption):
            pass

        with pytest.raises(UnsolvableError):
            generate_testsetup(Impl, ~Impl)
