import inspect
from dataclasses import replace

from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.execution.execution_flags import actor_only_backend_enabled
from ray.data._internal.logical.interfaces import (
    LogicalOperator,
    LogicalPlan,
    Plan,
    Rule,
)
from ray.data._internal.logical.operators import (
    Filter,
    FlatMap,
    MapBatches,
    MapRows,
    Project,
    Read,
    Write,
)
from ray.data.context import (
    DEFAULT_LARGE_ARGS_THRESHOLD,
    DEFAULT_SCHEDULING_STRATEGY,
    DEFAULT_SCHEDULING_STRATEGY_LARGE_ARGS,
)
from ray.data.datasource.planned_batch import SupportsReadBatchPlanning


class PlanReadBatches(Rule):
    """Let a cooperating source plan the first batch UDF's inputs."""

    def apply(self, plan: Plan) -> LogicalPlan:
        assert isinstance(plan, LogicalPlan)
        # The checkpoint planner can filter rows after the read. Its source and
        # sink must continue to see data rows for their existing key protocol.
        if (
            not plan.context.enable_planned_map_batches
            or plan.context.checkpoint_config is not None
        ):
            return plan

        # Validate the whole chain before changing the representation on an edge.
        # A join, a shared read in a union, or a limit may introduce a different
        # consumer or avoid reading whole batches.
        path = []
        op = plan.dag
        while isinstance(op, (MapBatches, MapRows, Filter, FlatMap, Project, Write)):
            path.append(op)
            (op,) = op.input_dependencies
        if not isinstance(op, Read) or not path or not isinstance(path[-1], MapBatches):
            return plan
        batch_map = path[-1]
        if op.planned_map_batch_size is not None:
            return plan
        if op.per_block_limit is not None or batch_map.per_block_limit is not None:
            return plan
        # Reading a descriptor's rows runs on the map worker. Keep explicit
        # read concurrency, worker isolation, placement and resource contracts.
        if (
            plan.context.isolate_read_workers
            or plan.context.scheduling_strategy != DEFAULT_SCHEDULING_STRATEGY
            or plan.context.scheduling_strategy_large_args
            != DEFAULT_SCHEDULING_STRATEGY_LARGE_ARGS
            or plan.context.large_args_threshold != DEFAULT_LARGE_ARGS_THRESHOLD
            or actor_only_backend_enabled(plan.context)
            or not isinstance(op.compute, TaskPoolStrategy)
            or op.compute.size is not None
            or not isinstance(batch_map.compute, TaskPoolStrategy)
            or batch_map.ray_remote_args_fn is not None
            or any(
                args not in ({}, {"scheduling_strategy": DEFAULT_SCHEDULING_STRATEGY})
                for args in (op.ray_remote_args, batch_map.ray_remote_args)
            )
        ):
            return plan
        if not inspect.isfunction(batch_map.fn) or (
            inspect.iscoroutinefunction(batch_map.fn)
            or inspect.isasyncgenfunction(batch_map.fn)
        ):
            return plan
        source = op.datasource_or_legacy_reader
        if (
            source is not op.datasource
            or not isinstance(source, SupportsReadBatchPlanning)
            or not source.supports_read_batch_planning()
        ):
            return plan

        size = batch_map.batch_size if batch_map.batch_size is not None else "auto"
        planned_read = replace(op, planned_map_batch_size=size)
        # These are descriptor rows. The physical map unwraps one descriptor
        # and invokes the original function on the corresponding source rows.
        child: LogicalOperator = replace(
            batch_map,
            input_dependencies=[planned_read],
            batch_size=1,
            min_rows_per_bundled_input=None,
        )
        for parent in reversed(path[:-1]):
            child = parent._with_new_input_dependencies([child])
        return LogicalPlan(child, plan.context)
