import threading
from typing import TYPE_CHECKING, List, Tuple

from ray.data._internal.planner import create_planner

if TYPE_CHECKING:
    from ray.data._internal.execution.execution_callback import ExecutionCallback

from .ruleset import Ruleset
from ray.data._internal.logical.interfaces import (
    LogicalPlan,
    Optimizer,
    PhysicalPlan,
    Rule,
)
from ray.data._internal.logical.operators.one_to_one_operator import _SchemaProbeLimit
from ray.data._internal.logical.rules import (
    BindGeneratedIdColumn,
    CombineDownloads,
    CombineShuffles,
    CommonSubExprElimination,
    ConfigureMapTaskMemoryUsingOutputSize,
    DeriveListFilesPushdown,
    EliminateAggregateShuffle,
    EnsureLocalAggregateReadsWholePartitions,
    ExtractEquiJoinKeys,
    FuseOperators,
    InheritTargetMaxBlockSizeRule,
    LimitPushdownRule,
    PredicatePushdown,
    ProjectionPushdown,
    PushdownCountFiles,
    SetReadParallelismRule,
    TopNRewriteRule,
)
from ray.data._internal.logical.rules.plan_read_batches import PlanReadBatches
from ray.util.annotations import DeveloperAPI

_LOGICAL_RULESET = Ruleset(
    [
        # Idempotent and independent of the rules below: it only moves
        # equalities from ``Join.condition`` into the join's condition keys.
        ExtractEquiJoinKeys,
        LimitPushdownRule,
        ProjectionPushdown,
        PredicatePushdown,
        CombineShuffles,
        CombineDownloads,
        PushdownCountFiles,
        TopNRewriteRule,
    ]
)

_PHYSICAL_RULESET = Ruleset(
    [
        InheritTargetMaxBlockSizeRule,
        SetReadParallelismRule,
        EliminateAggregateShuffle,
        FuseOperators,
        EnsureLocalAggregateReadsWholePartitions,
        ConfigureMapTaskMemoryUsingOutputSize,
    ]
)


_actor_only_rules_lock = threading.Lock()


def _sync_actor_only_rules() -> None:
    """Match the physical ruleset to whether the actor-only backend is enabled.

    This is the ruleset-swap wiring seam for the actor-only backend: when
    ``actor_only_backend_enabled()`` is true, the experimental ``FuseOperators``
    (which additionally allows Actor->Actor fusion) replaces the default rule in
    the physical ruleset; when it is false, the default rule is put back. The
    flag is read on every call, so turning the backend off restores the
    default ruleset.
    """
    from ray.data._internal.execution.execution_flags import actor_only_backend_enabled
    from ray.data._internal.experimental.logical.rules.operator_fusion import (
        FuseOperators as ExperimentalFuseOperators,
    )

    if actor_only_backend_enabled():
        old, new = FuseOperators, ExperimentalFuseOperators
    else:
        old, new = ExperimentalFuseOperators, FuseOperators

    # The check and the swap must run together: ``Ruleset.replace`` raises if
    # another thread already swapped ``old`` out.
    with _actor_only_rules_lock:
        if old in list(_PHYSICAL_RULESET):
            _PHYSICAL_RULESET.replace(old, new)


@DeveloperAPI
def get_logical_ruleset() -> Ruleset:
    return _LOGICAL_RULESET


@DeveloperAPI
def get_physical_ruleset() -> Ruleset:
    _sync_actor_only_rules()
    return _PHYSICAL_RULESET


class LogicalOptimizer(Optimizer):
    """The optimizer for logical operators."""

    @property
    def rules(self) -> List[Rule]:
        return [rule_cls() for rule_cls in get_logical_ruleset()]

    def _post_optimize(self, plan: LogicalPlan) -> LogicalPlan:
        # CommonSubExprElimination is only supposed to run once
        # isolated from the optimizer rule loop as it applies to
        # a single Projection operator not a chain of operators.
        plan = CommonSubExprElimination().apply(plan)
        # Runs after the rule loop so no ruleset rule, including ones callers
        # add, can project the generated ID column away again.
        plan = BindGeneratedIdColumn().apply(plan)
        # Must run last, over the final plan: it derives each ``ListFiles``'
        # listing-time constraints from the scanner of the ``ReadFiles`` that
        # consumes it. Running it earlier (or as a ruleset entry, which a
        # caller-added rule could follow) would let a later rewrite leave
        # ``ListFiles`` pruning by a predicate the reader no longer applies.
        return DeriveListFilesPushdown().apply(plan)


class PhysicalOptimizer(Optimizer):
    """The optimizer for physical operators."""

    @property
    def rules(self) -> List[Rule]:
        return [rule_cls() for rule_cls in get_physical_ruleset()]


def _optimize_for_execution(logical_plan: LogicalPlan) -> LogicalPlan:
    if isinstance(logical_plan.dag, _SchemaProbeLimit):
        # The UDF may need its planned range to produce even its first row.
        # Apply the probe limit after binding its inputs, so ordinary limit
        # pushdown cannot change the rows or identity seen during inference.
        (child,) = logical_plan.dag.input_dependencies
        child_plan = LogicalPlan(child, logical_plan.context)
        optimized_child = LogicalOptimizer().optimize(child_plan)
        planned_child = PlanReadBatches().apply(optimized_child)
        if planned_child is not optimized_child:
            probe = logical_plan.dag._with_new_input_dependencies([planned_child.dag])
            return LogicalPlan(probe, logical_plan.context)

    # 1. Logical -> Logical (Optimized)
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)

    # 2. Rewire Logical -> Logical (Optimized)
    logical_plan._dag = optimized_logical_plan.dag

    # Batch descriptors are private to this execution's Read -> MapBatches edge.
    # A Dataset derived from this one can have a different consumer or a shuffle.
    return PlanReadBatches().apply(optimized_logical_plan)


def get_execution_plan(
    logical_plan: LogicalPlan,
) -> Tuple[PhysicalPlan, List["ExecutionCallback"]]:
    """Optimize the logical plan, plan physical operators, then optimize them."""
    execution_plan = _optimize_for_execution(logical_plan)
    physical_plan, callbacks = create_planner().plan(execution_plan)
    return PhysicalOptimizer().optimize(physical_plan), callbacks
