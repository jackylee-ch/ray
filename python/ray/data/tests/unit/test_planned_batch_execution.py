import pyarrow as pa
import pytest

from ray.data._internal.planner.plan_udf_map_op import (
    _generate_transform_fn_for_map_batches,
    _map_planned_read_batches,
)
from ray.data.datasource.planned_batch import get_current_read_batch, range_blocks


class Source:
    def __init__(self, chunks):
        self.chunks = chunks

    def read_planned_batches(self, ranges):
        return (pa.table({"x": values}) for values in self.chunks)


def apply(fn, chunks, *, batch_format="pyarrow", zero_copy=True):
    transform = _map_planned_read_batches(
        _generate_transform_fn_for_map_batches(fn),
        Source(chunks).read_planned_batches,
        batch_format,
        zero_copy,
    )
    return transform(range_blocks("s", "v", "f", 3, 2), None)


@pytest.mark.parametrize("chunks", [[[0, 1]], [[0, 1], [2], [3]], [[0], [2]]])
def test_requires_one_complete_block_per_descriptor(chunks):
    with pytest.raises(ValueError):
        list(apply(lambda table: table, chunks))
    assert get_current_read_batch() is None


def test_range_context_ends_after_each_generator_step():
    def split(table):
        def results():
            for i in range(len(table)):
                span = get_current_read_batch()
                assert span is not None and span.start == table["x"][0].as_py()
                yield table.slice(i, 1)

        return results()

    batches = apply(split, [[0, 1], [2]])
    for value in range(3):
        assert next(batches)["x"].to_pylist() == [value]
        assert get_current_read_batch() is None
    assert list(batches) == []


def test_udf_exception_releases_range_context():
    def fail(table):
        span = get_current_read_batch()
        assert span is not None and span.start == 0
        raise RuntimeError("UDF failed")

    with pytest.raises(RuntimeError, match="UDF failed"):
        next(apply(fail, [[0, 1], [2]]))
    assert get_current_read_batch() is None


def test_mutable_numpy_batch_and_zero_output():
    def mutate(batch):
        batch["x"] += 10
        span = get_current_read_batch()
        assert span is not None
        if span.start == 0:
            return (item for item in ())
        return batch

    result = list(apply(mutate, [[0, 1], [2]], batch_format="numpy", zero_copy=False))
    assert len(result) == 1
    assert result[0]["x"].tolist() == [12]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
