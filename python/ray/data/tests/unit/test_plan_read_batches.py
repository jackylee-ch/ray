from dataclasses import replace
from unittest.mock import Mock

import pyarrow as pa
import pytest

from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators import (
    Count,
    Limit,
    MapBatches,
    MapRows,
    Read,
    Union,
    Write,
)
from ray.data._internal.logical.rules.plan_read_batches import PlanReadBatches
from ray.data.context import DataContext
from ray.data.datasource import Datasource
from ray.data.datasource.planned_batch import (
    ReadBatchRange,
    SupportsReadBatchPlanning,
    iter_planned_read_batches,
    iter_range_groups,
    range_blocks,
)


def identity(batch):
    return batch


class PlanningSource(Datasource, SupportsReadBatchPlanning):
    pass


@pytest.fixture
def read():
    source = Mock(spec=PlanningSource)
    source.supports_read_batch_planning.return_value = True
    return Read(source, source, parallelism=2)


@pytest.fixture
def context():
    return DataContext(enable_planned_map_batches=True)


@pytest.mark.parametrize(
    "batch_size, expected", [(1, 1), (64, 64), (None, "auto"), ("auto", "auto")]
)
def test_plans_only_first_udf_and_preserves_public_plan(
    read, context, batch_size, expected
):
    first = MapBatches(identity, input_dependencies=[read], batch_size=batch_size)
    second = MapBatches(identity, input_dependencies=[first], batch_size=7)
    write = Write(read.datasource, input_dependencies=[second])
    plan = LogicalPlan(write, context)
    result = PlanReadBatches().apply(plan)
    (
        planned_read,
        planned_first,
        planned_second,
        planned_write,
    ) = result.dag.post_order_iter()
    assert isinstance(planned_read, Read)
    assert isinstance(planned_first, MapBatches)
    assert isinstance(planned_second, MapBatches)
    assert isinstance(planned_write, Write)
    assert planned_read.planned_map_batch_size == expected
    assert planned_first.batch_size == 1
    assert planned_first.min_rows_per_bundled_input is None
    assert first.batch_size == batch_size
    assert planned_second.batch_size == 7
    assert (
        planned_write.datasink_or_legacy_datasource
        is write.datasink_or_legacy_datasource
    )
    assert read.planned_map_batch_size is None
    assert first.input_dependencies == [read]
    assert PlanReadBatches().apply(result) is result


@pytest.mark.parametrize(
    "parent",
    [
        lambda b: Union([b, b]),
        lambda b: Limit(1, input_dependencies=[b]),
        lambda b: Count(input_dependencies=[b]),
    ],
)
def test_skips_non_map_plan(read, context, parent):
    plan = LogicalPlan(parent(MapBatches(identity, input_dependencies=[read])), context)
    assert PlanReadBatches().apply(plan) is plan


def test_requires_direct_capable_source(read, context):
    plain = Mock(spec=Datasource)
    for source in (
        replace(read, datasource=plain, datasource_or_legacy_reader=plain),
        replace(read, datasource=plain),
        replace(read, per_block_limit=1),
        MapRows(identity, input_dependencies=[read]),
    ):
        plan = LogicalPlan(MapBatches(identity, input_dependencies=[source]), context)
        assert PlanReadBatches().apply(plan) is plan

    read.datasource.supports_read_batch_planning.return_value = False
    plan = LogicalPlan(MapBatches(identity, input_dependencies=[read]), context)
    assert PlanReadBatches().apply(plan) is plan


def test_does_not_enable_itself(read):
    ctx = DataContext()
    plan = LogicalPlan(MapBatches(identity, input_dependencies=[read]), ctx)
    assert PlanReadBatches().apply(plan) is plan
    ctx.enable_planned_map_batches = True
    ctx.checkpoint_config = {"id_column": "id", "checkpoint_path": "/checkpoint"}
    assert PlanReadBatches().apply(plan) is plan


def test_skips_deferred_udf_and_pushed_limit(read, context):
    async def generate(batch):
        yield batch

    async def transform(batch):
        return batch

    for op in (
        MapBatches(generate, input_dependencies=[read]),
        MapBatches(transform, input_dependencies=[read]),
        MapBatches(identity, input_dependencies=[read], per_block_limit=1),
    ):
        plan = LogicalPlan(op, context)
        assert PlanReadBatches().apply(plan) is plan


def test_respects_read_worker_requirements(read, context):
    def check(source, **map_args):
        op = MapBatches(identity, input_dependencies=[source], **map_args)
        plan = LogicalPlan(op, context)
        assert PlanReadBatches().apply(plan) is plan

    check(replace(read, ray_remote_args={"label_selector": {"ray.io/node-id": "head"}}))
    check(replace(read, compute=TaskPoolStrategy(size=2)))
    check(read, compute=ActorPoolStrategy(size=1))
    check(read, ray_remote_args={"num_gpus": 1})
    check(read, ray_remote_args_fn=lambda: {"runtime_env": {}})
    context._actor_only_backend_override = True
    check(read)
    context._actor_only_backend_override = False
    context.isolate_read_workers = True
    check(read)


@pytest.mark.parametrize(
    "option,value",
    [
        ("scheduling_strategy", "DEFAULT"),
        ("scheduling_strategy_large_args", "SPREAD"),
        ("large_args_threshold", 1),
    ],
)
def test_respects_context_scheduling(read, context, option, value):
    setattr(context, option, value)
    source = replace(read, ray_remote_args={"scheduling_strategy": "SPREAD"})
    plan = LogicalPlan(MapBatches(identity, input_dependencies=[source]), context)
    assert PlanReadBatches().apply(plan) is plan


def test_discovery_only_reads_descriptors(read, context):
    def never_run(table):
        raise AssertionError("Discovery executed the UDF")

    def metadata_task():
        yield from range_blocks("source", "version", "unit", 4, 3)

    source = read.datasource
    source.get_read_batch_tasks.return_value = [metadata_task]
    source.read_planned_batches.side_effect = AssertionError("Discovery read payload")
    dataset = Mock()
    dataset._logical_plan = LogicalPlan(
        MapBatches(never_run, input_dependencies=[read], batch_size=3), context
    )
    dataset._cache.get_bundle.return_value = None
    ranges = iter_planned_read_batches(dataset)
    source.get_read_batch_tasks.assert_not_called()
    assert list(ranges) == [
        ReadBatchRange("source", "version", "unit", 0, 3),
        ReadBatchRange("source", "version", "unit", 3, 4),
    ]
    source.get_read_batch_tasks.assert_called_once_with(1, 3, context)
    source.read_planned_batches.assert_not_called()

    dataset._cache.get_bundle.return_value = object()
    with pytest.raises(ValueError, match="does not execute source-planned batches"):
        list(iter_planned_read_batches(dataset))
    dataset._cache.get_bundle.return_value = None
    context.enable_planned_map_batches = False
    with pytest.raises(ValueError, match="does not execute source-planned batches"):
        list(iter_planned_read_batches(dataset))
    source.get_read_batch_tasks.assert_called_once()


def test_range_groups_bound_and_validate_the_transport():
    table = pa.concat_tables(list(range_blocks("source", "v", "0", 2050, 1)))
    groups = list(iter_range_groups([table]))
    assert [len(group) for group in groups] == [1024, 1024, 2]
    assert [span.start for group in groups for span in group] == list(range(2050))

    with pytest.raises(ValueError, match="RANGE_SCHEMA"):
        list(iter_range_groups([table.select(["start", "stop"])]))
    with pytest.raises(ValueError, match="RANGE_SCHEMA"):
        list(
            iter_range_groups(
                [table.set_column(4, "stop", table["stop"].cast(pa.int32()))]
            )
        )
    with pytest.raises(ValueError, match="cannot contain nulls"):
        list(
            iter_range_groups(
                [table.set_column(2, "unit", pa.nulls(2050, pa.string()))]
            )
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
