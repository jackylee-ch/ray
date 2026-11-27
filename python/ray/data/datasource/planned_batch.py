"""Source contracts for deterministic map input ranges."""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import pyarrow as pa

from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from ray.data.block import Block
    from ray.data.context import DataContext
    from ray.data.dataset import Dataset
    from ray.data.datasource.datasource import ReadTask


@DeveloperAPI
@dataclass(frozen=True)
class ReadBatchRange:
    """A half-open interval of source rows in a versioned unit, in scan order.

    Args:
        source: Identity of the source dataset.
        version: Immutable version of the source.
        unit: Source-defined unit identifier, independent of Ray parallelism.
        start: Inclusive logical row offset in the unit.
        stop: Exclusive logical row offset in the unit.
    """

    source: str
    version: str
    unit: str
    start: int
    stop: int

    def __post_init__(self):
        if not 0 <= self.start < self.stop:
            raise ValueError("A planned source batch must cover a nonempty row range")


_current_range: ContextVar[Optional[ReadBatchRange]] = ContextVar(
    "ray_data_read_batch_range", default=None
)


@DeveloperAPI
def get_current_read_batch() -> Optional[ReadBatchRange]:
    """Return the input range inside a source-planned batch UDF, or None."""
    return _current_range.get()


RANGE_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("version", pa.string()),
        ("unit", pa.string()),
        ("start", pa.int64()),
        ("stop", pa.int64()),
    ]
)
_MAX_RANGES_PER_GROUP = 1024


def iter_range_groups(tables: Iterable[pa.Table]) -> Iterator[List[ReadBatchRange]]:
    """Validate descriptor tables, retaining at most one group of decoded rows."""
    for table in tables:
        if not isinstance(table, pa.Table) or not table.schema.equals(RANGE_SCHEMA):
            raise ValueError("Expected a source range table with RANGE_SCHEMA")
        if any(column.null_count for column in table.columns):
            raise ValueError("A source range table cannot contain nulls")
        for offset in range(0, table.num_rows, _MAX_RANGES_PER_GROUP):
            group = table.slice(offset, _MAX_RANGES_PER_GROUP)
            columns = (column.to_pylist() for column in group.columns)
            yield [ReadBatchRange(*values) for values in zip(*columns)]


def range_blocks(
    source: str,
    version: str,
    unit: str,
    num_rows: int,
    batch_size: int,
    *,
    start: int = 0,
) -> Iterator[pa.Table]:
    """Emit bounded descriptor tables for [start, num_rows) on the unit's grid."""
    if batch_size <= 0 or not 0 <= start <= num_rows or start % batch_size:
        raise ValueError("Invalid source batch size or range start")
    # Group small descriptors into transport blocks; a row is one UDF input.
    group_size = batch_size * _MAX_RANGES_PER_GROUP
    for first in range(start, num_rows, group_size):
        offsets = range(first, min(first + group_size, num_rows), batch_size)
        yield pa.Table.from_arrays(
            [
                pa.repeat(source, len(offsets)),
                pa.repeat(version, len(offsets)),
                pa.repeat(unit, len(offsets)),
                pa.array(offsets, type=pa.int64()),
                pa.array(
                    [min(i + batch_size, num_rows) for i in offsets], type=pa.int64()
                ),
            ],
            schema=RANGE_SCHEMA,
        )


@DeveloperAPI
class SupportsReadBatchPlanning(ABC):
    """Optional datasource capability for replayable input ranges.

    A scan must have pinned versions, deterministic row order and the same ranges
    for every parallelism setting. The source must decline unsupported options.
    """

    @abstractmethod
    def supports_read_batch_planning(self) -> bool:
        """Whether this particular scan can satisfy the range contract."""

    @abstractmethod
    def get_read_batch_tasks(
        self,
        parallelism: int,
        batch_size: Union[int, Literal["auto"]],
        data_context: "DataContext",
    ) -> List["ReadTask"]:
        """Plan metadata-only tasks yielding Arrow tables with RANGE_SCHEMA.

        The task callable must be safe to iterate on the driver without loading
        payload rows or accessing worker-only resources. Intervals uniquely cover
        the selected scan, without gaps or overlaps, at every parallelism.

        Args:
            parallelism: Requested parallelism for distributing descriptors.
            batch_size: Rows per range within each unit, or source-estimated sizing.
            data_context: Execution options for this scan.

        Returns:
            Metadata-only read tasks covering the selected source snapshot.
        """

    @abstractmethod
    def read_planned_batches(
        self, ranges: Sequence[ReadBatchRange]
    ) -> Iterator["Block"]:
        """Read one block per range, in order, using the scan's original projection.

        Each range must replay the same rows when requested alone, with other
        ranges or on a different worker.

        Args:
            ranges: Ordered nonempty ranges. Adjacent ranges may share a scan.

        Returns:
            An iterator over the payload blocks in the requested range order.
        """

    def get_read_batch_reader(
        self,
    ) -> Callable[[Sequence[ReadBatchRange]], Iterator["Block"]]:
        """Return a serializable callable that reads the given ranges in order.

        Override this method when planning state is costly to ship to workers.
        """
        return self.read_planned_batches


@DeveloperAPI
def iter_planned_read_batches(dataset: "Dataset") -> Iterator[ReadBatchRange]:
    """Iterate over the expected inputs of the first source-planned batch UDF.

    Enumeration uses the given dataset's pinned source and current execution
    options. It reads source metadata on the driver without executing the UDF
    or fetching payload rows. Task concurrency and completion order do not
    affect the set of ranges. A task retry can execute a range more than once.

    Args:
        dataset: The unmaterialized map-only dataset to inspect. Enable
            ``DataContext.enable_planned_map_batches`` before creating it.

    Yields:
        ReadBatchRange: The source interval of one expected UDF invocation.

    Raises:
        ValueError: If this dataset would not execute a source-planned batch UDF.
    """
    from ray.data._internal.logical.operators import Read
    from ray.data._internal.logical.optimizers import LogicalOptimizer
    from ray.data._internal.logical.rules.plan_read_batches import PlanReadBatches

    logical = dataset._logical_plan
    if (
        not logical.context.enable_planned_map_batches
        or dataset._cache.get_bundle(logical.dag) is not None
    ):
        raise ValueError("This dataset does not execute source-planned batches")
    optimized = LogicalOptimizer().optimize(logical)
    planned = PlanReadBatches().apply(optimized)
    if planned is optimized:
        raise ValueError("This dataset does not execute source-planned batches")

    read = next(planned.dag.post_order_iter())
    assert isinstance(read, Read) and read.planned_map_batch_size is not None
    source = read.datasource
    assert isinstance(source, SupportsReadBatchPlanning)
    # A provider guarantees the same grid for every parallelism value.
    for task in source.get_read_batch_tasks(
        1, read.planned_map_batch_size, logical.context
    ):
        for group in iter_range_groups(task()):
            yield from group
