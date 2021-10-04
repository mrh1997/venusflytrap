import pytest
from unittest.mock import patch
from venusflytrap import (
    TestOption,
    Constraint,
    No,
    Any,
    All,
    MaxOne,
    ExactOne,
    Implies,
    Equal,
    Type,
    requires,
    link,
    bind,
    bind_optional,
    bind_set,
    UnsolvableError,
    TestSetup,
    TestHandler,
    handler,
    avail_impls,
    WeightFunc,
)
import z3  # type: ignore
from typing import List, Callable, cast


# provide tbe following class tree as fixtures:
# Base
# +- Impl1
# +- Impl2
# +- Impl3


@pytest.fixture
def Base() -> Type[TestOption]:
    class Base(TestOption, group=True):
        pass

    return Base


@pytest.fixture
def Impl1(Base) -> Type[TestOption]:
    class Impl1(Base):
        pass

    return Impl1


@pytest.fixture
def Impl2(Base) -> Type[TestOption]:
    class Impl2(Base):
        pass

    return Impl2


@pytest.fixture
def Impl3(Base) -> Type[TestOption]:
    class Impl3(Base):
        pass

    return Impl3


@pytest.fixture
def ImplRoot(Base):
    class ImplRoot(TestOption):
        dep = bind_optional(Base)

    return ImplRoot


@pytest.fixture
def AltBase() -> Type[TestOption]:
    class Base_2(TestOption, group=True):
        pass

    return Base_2


@pytest.fixture
def AltImpl1(AltBase) -> Type[TestOption]:
    class AltImpl1(AltBase):
        pass

    return AltImpl1


@pytest.fixture
def AltImpl2(AltBase) -> Type[TestOption]:
    class AltImpl2(AltBase):
        pass

    return AltImpl2


def assert_constr(*constraints, expect=None):
    s = z3.Solver()
    s.add(*[c._to_z3_formula() for c in constraints])
    assert s.check() == z3.sat, "Constraint is not satisfiable"
    m = s.model()
    for impl, expected_state in (expect or {}).items():
        err_text = f"'{impl!r}' is {not expected_state} instead of {expected_state}"
        assert m[impl._to_z3_formula()] == expected_state, err_text


def assert_constr_fail(*constraints):
    s = z3.Solver()
    s.add(*[c._to_z3_formula() for c in constraints])
    assert s.check() == z3.unsat, "Constraint is satisfiable"


class TestConstraint:

    SETOPS: List[Callable] = [No, Any, All, ExactOne, MaxOne]

    def test_toZ3Formula_onGroupClass_raisesValueError(self, Base):
        with pytest.raises(ValueError):
            Base._to_z3_formula()

    def test_toZ3Formula_onImplementation_returnsBoolRef(self, Impl1):
        assert isinstance(Impl1._to_z3_formula(), z3.BoolRef)

    def test_Not(self, Impl1):
        assert_constr(~Impl1, expect={Impl1: False})
        assert_constr_fail(~Impl1, Impl1)

    def test_And(self, Impl1, Impl2):
        assert_constr(Impl1 & Impl2, expect={Impl1: True, Impl2: True})
        assert_constr_fail(Impl1 & Impl2, ~Impl1)

    def test_Or(self, Impl1, Impl2):
        assert_constr(Impl1 | Impl2, Impl1, Impl2)
        assert_constr(Impl1 | Impl2, ~Impl1, expect={Impl2: True})
        assert_constr(Impl1 | Impl2, ~Impl2, expect={Impl1: True})
        assert_constr_fail(Impl1 | Impl2, ~Impl1, ~Impl2)

    def test_Xor(self, Impl1, Impl2):
        assert_constr(Impl1 ^ Impl2, ~Impl1, expect={Impl2: True})
        assert_constr(Impl1 ^ Impl2, Impl1, expect={Impl2: False})
        assert_constr_fail(Impl1 ^ Impl2, ~Impl1, ~Impl2)
        assert_constr_fail(Impl1 ^ Impl2, Impl1, Impl2)

    def test_Implies(self, Impl1, Impl2):
        assert_constr(Implies(Impl1, Impl2), ~Impl1, Impl2)
        assert_constr(Implies(Impl1, Impl2), ~Impl1, ~Impl2)
        assert_constr(Implies(Impl1, Impl2), Impl1, expect={Impl2: True})
        assert_constr_fail(Implies(Impl1, Impl2), Impl1, ~Impl2)

    def test_Linked(self, Impl1, Impl2):
        assert_constr(Equal(Impl1, Impl2), ~Impl1, expect={Impl2: False})
        assert_constr(Equal(Impl1, Impl2), Impl1, expect={Impl2: True})
        assert_constr_fail(Equal(Impl1, Impl2), ~Impl1, Impl2)

    def test_iter_onTestOption_returnsSelf(self, Base, Impl1):
        assert set(iter(Base)) == {Base}
        assert set(iter(Impl1)) == {Impl1}

    def test_iter_onLogicalOperators_returnsOperands(self, Impl1, Impl2):
        assert set(iter(Impl1 & ~Impl2)) == {Impl1, Impl2}
        assert set(iter(Implies(Impl1, Impl2))) == {Impl1, Impl2}
        assert set(iter(Equal(Impl1, Impl2))) == {Impl1, Impl2}

    @pytest.mark.parametrize("operator", SETOPS)
    def test_iter_onSetOps_returnsAllImplementations(
        self, Base, Impl1, Impl2, operator
    ):
        assert set(operator(Base)) == {Impl1, Impl2}

    @pytest.mark.parametrize("operator", SETOPS)
    def test_iter_onSetOpsWithExcludingFilter_returnsAllImplementationsExceptSpecifiedOnes(
        self, Base, Impl1, Impl2, Impl3, operator
    ):
        assert set(operator(Base, excluding={Impl2, Impl3})) == {Impl1}

    @pytest.mark.parametrize("operator", SETOPS)
    def test_iter_onSetOpsWithWhereFilter_returnsAllImplementationsMatchingWhereFunc(
        self, Base, Impl1, Impl2, Impl3, operator
    ):
        assert set(operator(Base, where=lambda i: i.__name__ >= "Impl2")) == {
            Impl2,
            Impl3,
        }

    def test_No(self, Base, Impl1, Impl2, Impl3):
        assert_constr(No(Base), expect={Impl1: False, Impl2: False, Impl3: False})
        assert_constr_fail(No(Base), Impl2)

    def test_Any(self, Base, Impl1, Impl2, Impl3):
        assert_constr(Any(Base), ~Impl1, ~Impl3, expect={Impl2: True})
        assert_constr(Any(Base), Impl1, Impl2)
        assert_constr_fail(Any(Base), ~Impl1, ~Impl2, ~Impl3)

    def test_All(self, Base, Impl1, Impl2, Impl3):
        assert_constr(All(Base), expect={Impl1: True, Impl2: True, Impl3: True})
        assert_constr_fail(All(Base), ~Impl1)

    def test_ExactOne(self, Base, Impl1, Impl2, Impl3):
        assert_constr(ExactOne(Base), Impl1, expect={Impl2: False, Impl3: False})
        assert_constr(ExactOne(Base), ~Impl1, ~Impl2, expect={Impl3: True})
        assert_constr_fail(ExactOne(Base), Impl1, Impl2)

    def test_ExactOne_onNoImplementations_fails(self, Base):
        assert_constr_fail(ExactOne(Base))

    def test_MaxOne(self, Base, Impl1, Impl2, Impl3):
        assert_constr(MaxOne(Base), Impl1, expect={Impl2: False, Impl3: False})
        assert_constr(MaxOne(Base), ~Impl1, ~Impl2, ~Impl3)
        assert_constr_fail(MaxOne(Base), Impl1, Impl2)

    def test_MaxOne_onNoImplementations_returnsTrue(self, Base):
        assert_constr(MaxOne(Base))

    @pytest.mark.parametrize("operator", SETOPS)
    def test_SetOps_onMultipleCalls_optimizeByCachingResult(self, Impl1, operator):
        assert operator(Impl1)._to_z3_formula() is operator(Impl1)._to_z3_formula()


class Test_TestOption:
    def test_create_onGroupWithoutImplementations_setImplementationsEmpty(self, Base):
        assert Base.implementations == set()

    def test_create_onImplementation_setImplementationsToSelf(self, Impl1):
        assert Impl1.implementations == {Impl1}

    def test_create_onMultipleGroupBases_addsSelfToBasesImplementationLists(self, Base):
        class Base2(TestOption, group=True):
            pass

        class Impl(Base, Base2):
            pass

        assert all(cls.implementations == {Impl} for cls in [Base, Base2])

    def test_create_onMulipleImplementations_addsAllToBaseClassImplementations(
        self, Base, Impl1, Impl2
    ):
        assert Base.implementations == {Impl1, Impl2}

    def test_create_onBaseHierarchy_maintainsDifferentImplPerBase(self, Base, Impl1):
        class Base2(Base, group=True):
            pass

        class Impl21(Base2):
            pass

        assert Base.implementations == {Impl1, Impl21}
        assert Base2.implementations == {Impl21}

    def test_create_onImplementation_callsRegisterImplementionOnAllParentClasses(self):
        class Base(TestOption, group=True):
            registered_children = set()

            @classmethod
            def register_implementation(cls, impl_cls: Type[TestOption]):
                super().register_implementation(impl_cls)
                cls.registered_children.add(impl_cls)

        class Base2(Base, group=True):
            pass

        class Impl21(Base2):
            pass

        class Impl22(Base2):
            pass

        assert Base.registered_children == {Impl21, Impl22}

    def test_create_onBindings_addsConstraints(self, Base, AltBase):
        class Alt2Base(TestOption, group=True):
            pass

        class Impl(TestOption):
            dep = bind(Base)
            opt_dep = bind_optional(AltBase)
            set_dep = bind_set(Alt2Base)

        assert set(Impl.constraints) == {
            Implies(Impl, ExactOne(Base)),
            Implies(Impl, No(AltBase) | ExactOne(AltBase)),
        }

    def test_create_onWrongAttrType_raiseMeaningfulTypeError(self):
        with pytest.raises(TypeError) as exc:

            class Impl(TestOption):
                dep = bind(cast(Type[TestOption], int))

        assert "Impl.dep" in str(exc)

    def test_create_onHandlers_setsAttrsToNone(self):
        class MyHandler(TestHandler):
            pass

        class Impl(TestOption):
            hndl = handler(MyHandler)

        assert Impl.hndl is None

    def test_instanciate_onGroupType_raisesNotImplementedError(self, Base):
        with pytest.raises(NotImplementedError):
            Base()

    def test_iterDependencies_onImplType_returnsOnlySelf(self, Impl1):
        assert set(Impl1.iter_dependencies()) == {Impl1}

    def test_iterDependencies_onGroupType_returnsEmpty(self, Base):
        assert set(Base.iter_dependencies()) == set()

    def test_iterDependencies_onRecursiveDepenedencies_ok(self, Base, Impl1):
        class Base2(TestOption, group=True):
            pass

        class Impl2(Base2):
            dep = bind(Base)

        class Impl3(TestOption):
            dep = bind(Base2)

        assert set(Impl3.iter_dependencies()) == {Impl1, Impl2, Impl3}

    def test_iterDependencies_onOptionalAndSetAttrs_ok(
        self, Base, Impl1, AltBase, AltImpl1
    ):
        class RootImpl(TestOption):
            opt_dep = bind_optional(Base)
            set_dep = bind_set(AltBase)

        assert set(RootImpl.iter_dependencies()) == {Impl1, AltImpl1, RootImpl}

    def test_iterDependencies_onReferredByConstraintsOnly_addedToList(self, Impl1):
        class Impl2(TestOption):
            dep = bind(Impl1)

        @requires(Impl2)
        class Impl3(TestOption):
            pass

        assert set(Impl3.iter_dependencies()) == {Impl1, Impl2, Impl3}

    def test_iterDependencies_orderedFromLeafToRoot(self, Impl1):
        class ImplMiddle1(TestOption):
            dep = bind(Impl1)

        class ImplMiddle2(TestOption):
            dep = bind(Impl1)

        class ImplRoot(TestOption):
            dep1 = bind(ImplMiddle2)
            dep2 = bind(ImplMiddle1)

        leaf, *middle, root = list(ImplRoot.iter_dependencies())
        assert leaf == Impl1
        assert set(middle) == {ImplMiddle1, ImplMiddle2}
        assert root == ImplRoot

    def test_requiresDecorator_onGroupClass_addsConstraintsToImpls(self, Base):
        base_constr, impl1_constr = Constraint(), Constraint()
        requires(base_constr, by=Base)

        @requires(impl1_constr)
        class Impl1(Base):
            pass

        assert set(Base.constraints) == {Implies(Base, base_constr)}
        assert set(Impl1.constraints) == {
            Implies(Base, base_constr),
            Implies(Impl1, impl1_constr),
        }

    def test_linksDecorator_onGroupClass_addsConstraintsToImpls(self, Base):
        base_constr, impl1_constr = Constraint(), Constraint()
        link(base_constr, Base)

        @link(impl1_constr)
        class Impl1(Base):
            pass

        assert set(Base.constraints) == {Equal(Base, base_constr)}
        assert set(Impl1.constraints) == {
            Equal(Base, base_constr),
            Equal(Impl1, impl1_constr),
        }

    def test_constraintDecorator_onImpl_doesNotModifyConstraintsOfSibling(
        self, Base, Impl1, Impl2
    ):
        requires(Constraint(), by=Impl1)
        assert len(Impl2.constraints) == 0

    def test_availImpls_onBoundType_returnsAllImpls(self, Base, Impl1, Impl2):
        class UserOfBase(TestOption):
            dep = bind(Base)
            dep_optional = bind_optional(Base)
            dep_set = bind_set(Base)

        assert avail_impls(UserOfBase.dep) == {Impl1, Impl2}
        assert avail_impls(UserOfBase.dep_optional) == {Impl1, Impl2}
        assert avail_impls(UserOfBase.dep_set) == {Impl1, Impl2}

    def test_getAbsoluteWeights_onImplWithoutExplicitWeight_returnsWeight1(self, Impl1):
        abs_weights = Impl1.get_absolute_weights()
        assert abs_weights[Impl1] == 1.0

    def test_getAbsoluteWeights_onGroupWithoutExplicitWeight_returnsSumOfChildWeights(
        self, Base, Impl1, Impl2, Impl3
    ):
        abs_weights = Base.get_absolute_weights()
        assert abs_weights == {Base: 3.0, Impl1: 1.0, Impl2: 1.0, Impl3: 1.0}

    def test_getAbsoluteWeights_onWeightSpecifiedAsClassParameter_returnsClassParameter(
        self,
    ):
        class Impl(TestOption, weight=3.21):
            pass

        abs_weights = Impl.get_absolute_weights()
        assert abs_weights == {Impl: 3.21}

    def test_getAbsoluteWeights_onWeightSpecifiedAsWeightFunc_overridesClassParameter(
        self,
    ):
        class Impl(TestOption, weight=3):
            pass

        weight_func = cast(WeightFunc, {Impl: 4.0}.get)
        abs_weights = Impl.get_absolute_weights(weight_func=weight_func)
        assert abs_weights == {Impl: 4}

    def test_getAbsoluteWeights_onGroupHasWeight_scalesChildrenWeightSumToGroupWeight(
        self, Base, Impl1, Impl2
    ):
        weight_func = {Base: 100, Impl1: 2, Impl2: 8}.get
        abs_weights = Base.get_absolute_weights(weight_func=weight_func)
        assert abs_weights == {Base: 100, Impl1: 20, Impl2: 80}

    def test_getAbsoluteWeights_onGroupHasWeight_scalesOverMultipleLevels(self, Base):
        class SubBase(Base, group=True):
            pass

        class Impl(SubBase):
            pass

        abs_weights = Base.get_absolute_weights(weight_func={Base: 10}.get)
        assert abs_weights == {Base: 10, SubBase: 10, Impl: 10}

    def test_getAbsoluteWeights_onGroupHasWeight_usesWeightForParentGroup(self, Base):
        class SubBase(Base, group=True):
            pass

        class Impl(SubBase):
            pass

        weight_func = cast(WeightFunc, {SubBase: 10}.get)
        abs_weights = Base.get_absolute_weights(weight_func=weight_func)
        assert abs_weights == {Base: 10, SubBase: 10, Impl: 10}

    def test_getAbsoluteWeights_onMultipleParentGroups_rescaleRecursivly(self, Base):
        class SubBase(Base, group=True):
            pass

        class ImplSub(Base):
            pass

        class Impl(SubBase):
            pass

        weight_func = cast(WeightFunc, {Base: 100, SubBase: 9, ImplSub: 1}.get)
        abs_weights = Base.get_absolute_weights(weight_func=weight_func)
        assert abs_weights[Impl] == 90


class TestGenerateTestSetup:
    def test_generateTestsetup_onImpl_instanciatesTestSetup(self):
        class Impl(TestOption):
            def __init__(self):
                super().__init__()
                self.val = 123

        ts = Impl.generate_testsetup()
        assert isinstance(ts, TestSetup)
        assert ts.root_inst.val == 123

    def test_generateTestsetup_onGroup_raisesUnsolvableError(self, Base):
        with pytest.raises(UnsolvableError):
            _ = Base.generate_testsetup()

    def test_generateTestsetup_instanciatesAndLinksDependendImpls(self):
        class Base(TestOption, group=True):
            val = 1

        class Impl1(Base):
            def __init__(self):
                super().__init__()
                self.val = 123

        class Impl2(TestOption):
            dep = bind(Base)

        ts = Impl2.generate_testsetup().root_inst
        assert ts.dep.val == 123

    def test_generateTestsetup_onAdditionalConstraints_instantiatesImpls(
        self, Base, Impl1, Impl2, Impl3
    ):
        def base_init(self):
            nonlocal inst_count
            super(Base, self).__init__()
            inst_count += 1

        Base.__init__ = base_init
        inst_count = 0
        Impl1.generate_testsetup(~Impl2, Impl3)
        assert inst_count == 2

    def test_generateTestsetup_onRootUnavailable_raisesUnsolvableError(self, Impl1):
        with pytest.raises(UnsolvableError):
            _ = Impl1.generate_testsetup(~Impl1)

    def test_generateTestsetup_onOptionalDepAndNoImpl_setsDepToNone(self, Base):
        class Impl(TestOption):
            dep = bind_optional(Base)

        ts = Impl.generate_testsetup().root_inst
        assert ts.dep is None

    @pytest.mark.parametrize("impl_count", [0, 1, 2])
    def test_generateTestsetup_onSetDep_returnsSet(self, impl_count, Base):
        impls = {type("Impl", (Base,), {}) for c in range(impl_count)}

        class ImplDep(TestOption):
            dep = bind_set(Base)

        ts = ImplDep.generate_testsetup(*impls).root_inst
        assert {type(to) for to in ts.dep} == impls

    def test_generateTestsetup_onSelectExactOneImpl_ok(self, Base, ImplRoot):
        def impl_init(self):
            nonlocal inst_cnt
            inst_cnt += 1

        impls = [type("Impl", (Base,), {"__init__": impl_init}) for c in range(10)]

        inst_cnt = 0
        ts = ImplRoot.generate_testsetup().root_inst
        assert isinstance(ts.dep, tuple(impls))
        assert inst_cnt == 1

    @pytest.mark.parametrize("high_weight_testoption_ndx", [0, 1, 2])
    def test_generateTestsetup_usesWeightsFromGetAbsoluteWeights(
        self, ImplRoot, Impl1, Impl2, Impl3, high_weight_testoption_ndx
    ):
        impls = [Impl1, Impl2, Impl3]
        high_weight_testoption = impls[high_weight_testoption_ndx]
        weight_map = dict.fromkeys(impls, 1.0)
        weight_map[high_weight_testoption] = 2.0
        with patch.object(
            TestOption,
            "get_absolute_weights",
            return_value=weight_map,
        ) as get_absolute_weights_mock:
            wf = lambda i: 1
            ts = ImplRoot.generate_testsetup(ImplRoot, ImplRoot, weight_func=wf)
        get_absolute_weights_mock.assert_called_with(ImplRoot, ImplRoot, weight_func=wf)
        assert isinstance(ts.root_inst.dep, cast(type, high_weight_testoption))

    def test_generateTestsetup_onWeightIsTooSmall_returnsValueError(
        self, ImplRoot, Impl1
    ):
        with pytest.raises(ValueError):
            too_low_weight = {Impl1: 0.0000001}.get
            ts = ImplRoot.generate_testsetup(weight_func=too_low_weight)

    def test_generateTestsetup_onUnsolvable_raisesUnsolvableError(self, Impl1):
        with pytest.raises(UnsolvableError):
            Impl1.generate_testsetup(~Impl1)

    def test_run_returnsInstanceOfRoot(self, Impl1):
        root_inst = Impl1.generate_testsetup().run()
        assert isinstance(root_inst, Impl1)

    def test_run_callsSetupOfChildrenThenRunThenTearDownOfChildren(self):
        class Base(TestOption, group=True):
            def setup(self):
                call_order.append(f"SETUP {self.__class__.__name__}")

            def test(self):
                call_order.append(f"RUN {self.__class__.__name__}")

            def teardown(self):
                call_order.append(f"TEARDOWN {self.__class__.__name__}")

        class ImplChild(Base):
            pass

        class ImplRoot(Base):
            dep = bind(ImplChild)

        call_order = []
        ImplRoot.generate_testsetup().run()
        assert call_order == [
            "SETUP ImplChild",
            "SETUP ImplRoot",
            "RUN ImplRoot",
            "TEARDOWN ImplRoot",
            "TEARDOWN ImplChild",
        ]

    def test_run_onDisabledImpls_callsSetupOfEnabledImplsOnly(self):
        class Base(TestOption, group=True):
            def setup(self):
                call_order.append(f"SETUP {self.__class__.__name__}")

        class ImplChild(Base):
            pass

        @requires(~ImplChild)
        class ImplRoot(Base):
            dep = bind_optional(ImplChild)

        call_order = []
        ImplRoot.generate_testsetup().run()
        assert call_order == ["SETUP ImplRoot"]

    def test_run_onMultipleRefsToChild_callsSetupOnlyOnce(self):
        class Base(TestOption, group=True):
            def setup(self):
                call_order.append(f"SETUP {self.__class__.__name__}")

        class ImplChild2(Base):
            pass

        class ImplChild11(Base):
            dep = bind(ImplChild2)

        class ImplChild12(Base):
            dep = bind(ImplChild2)

        class ImplRoot(Base):
            dep1 = bind(ImplChild11)
            dep2 = bind(ImplChild12)

        call_order = []
        ImplRoot.generate_testsetup().run()
        assert call_order[0] == "SETUP ImplChild2"
        assert call_order[-1] == "SETUP ImplRoot"

    def test_run_onHandlers_setsAttrsToPassedHandlers(self):
        class MyHandler1(TestHandler):
            pass

        class MyHandler2(TestHandler):
            pass

        class Impl(TestOption):
            handler1 = handler(MyHandler1)
            handler2 = handler(MyHandler2)

        ts = Impl.generate_testsetup()
        handlers = {MyHandler1(), MyHandler2()}
        result = ts.run(*handlers)
        assert {result.handler1, result.handler2} == handlers

    def test_run_onHandlerInDependency_setsHandlerAttrOfDependency(self):
        class MyHandler(TestHandler):
            pass

        class ChildImpl(TestOption):
            handler = handler(MyHandler)

        class Impl(TestOption):
            dep = bind(ChildImpl)

        ts = Impl.generate_testsetup()
        myhandler = MyHandler()
        result = ts.run(myhandler)
        assert result.dep.handler is myhandler

    def test_run_onPassingMultipleHandlersOfSameClass_raisesValueError(self):
        class MyHandler(TestHandler):
            pass

        class Impl(TestOption):
            hndl = handler(MyHandler)

        with pytest.raises(ValueError):
            Impl.generate_testsetup().run(MyHandler(), MyHandler())

    def test_run_onNonHandlerParameter_raisesTypeError(self):
        with pytest.raises(TypeError):

            class Impl(TestOption):
                hndl = handler(cast(Type[TestHandler], int))

    def test_availImpls_onInstance_raisesValuError(self):
        class Impl(TestOption):
            pass

        class UserOfImpl(TestOption):
            dep = bind(Impl)
            dep_optional = bind_optional(Impl)
            dep_set = bind_set(Impl)

        ts = UserOfImpl.generate_testsetup()
        with pytest.raises(ValueError):
            avail_impls(ts.root_inst)
        with pytest.raises(ValueError):
            avail_impls(ts.root_inst)
        with pytest.raises(ValueError):
            avail_impls(ts.root_inst)
