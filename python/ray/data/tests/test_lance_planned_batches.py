import pickle
from dataclasses import replace

import pyarrow as pa
import pytest

import ray
from ray.data._internal.datasource.lance_datasource import LanceDatasource
from ray.data._internal.logical.operators import MapBatches, Read
from ray.data.context import DataContext
from ray.data.datasource.planned_batch import ReadBatchRange

lance = pytest.importorskip("lance")


def _plan(source, batch_size, parallelism):
    tasks = source.get_read_batch_tasks(
        parallelism, batch_size, DataContext.get_current()
    )
    ranges = []
    for task in tasks:
        descriptors = [
            ReadBatchRange(**r) for block in task() for r in block.to_pylist()
        ]
        assert len(descriptors) == task.metadata.num_rows
        ranges.extend(descriptors)
    return ranges


def test_planned_lance_batch_snapshot_projection_and_live_offsets(tmp_path):
    uri = str(tmp_path / "ds.lance")
    ds = lance.write_dataset(
        pa.table({"id": range(19), "ignored": range(19)}),
        uri,
        max_rows_per_file=8,
        max_rows_per_group=3,
    )
    ds.delete("id IN (1, 3, 9)")
    source = LanceDatasource(uri, columns=["id"], scanner_options={"batch_size": 2})
    assert source.supports_read_batch_planning()
    ranges = _plan(source, 4, 1)
    assert ranges == _plan(source, 4, 7)
    assert [(r.unit, r.start, r.stop) for r in ranges] == [
        ("0", 0, 4),
        ("0", 4, 6),
        ("1", 0, 4),
        ("1", 4, 7),
        ("2", 0, 3),
    ]
    assert {r.version for r in ranges} == {str(ds.version)}
    ds.delete("id = 2")
    # A worker reopens the pinned snapshot after a newer version is published.
    restored = pickle.loads(pickle.dumps(source))
    result = [t.to_pydict() for t in restored.read_planned_batches(ranges)]
    assert result == [
        {"id": [0, 2, 4, 5]},
        {"id": [6, 7]},
        {"id": [8, 10, 11, 12]},
        {"id": [13, 14, 15]},
        {"id": [16, 17, 18]},
    ]
    # A retry may replay a non-adjacent subset, including ranges out of order.
    subset = [ranges[3], ranges[1], ranges[4]]
    assert [t.to_pydict() for t in restored.read_planned_batches(subset)] == [
        result[3],
        result[1],
        result[4],
    ]


def test_planned_lance_batch_auto_multisource(tmp_path, monkeypatch):
    from ray.data._internal.datasource import lance_datasource

    uris = [str(tmp_path / name) for name in ["one.lance", "two.lance"]]
    for uri, length in zip(uris, [21, 8]):
        lance.write_dataset(pa.table({"id": range(length)}), uri)
    ds = LanceDatasource(uris)
    monkeypatch.setattr(lance_datasource, "DEFAULT_BATCH_SIZE_BYTES", 128)
    ranges = _plan(ds, "auto", 1)
    assert ranges == _plan(ds, "auto", 6)
    assert {r.source for r in ranges} == {d.uri for d in ds.lance_datasets}
    assert sum(r.stop - r.start for r in ranges) == 29
    assert [
        row
        for batch in ds.read_planned_batches(ranges)
        for row in batch["id"].to_pylist()
    ] == [*range(21), *range(8)]
    with pytest.raises(ValueError, match="snapshot"):
        list(ds.read_planned_batches([replace(ranges[0], version="invalid")]))
    with pytest.raises(ValueError, match="exceeds"):
        list(ds.read_planned_batches([replace(ranges[0], stop=22)]))
    with pytest.raises(ValueError, match="fragment"):
        list(ds.read_planned_batches([replace(ranges[0], unit="9999")]))


def test_planned_lance_mid_scan_retry(fragmented_data, monkeypatch):
    from ray.data._internal.datasource import lance_datasource

    path, values, _ = fragmented_data
    source = LanceDatasource(path)
    source._retry_params.update(match=["transient scan"], max_backoff_s=0)
    ranges = _plan(source, 3, 1)
    scan = lance_datasource._scan_range_span
    first_attempt = True

    def read(*args):
        nonlocal first_attempt
        batches = scan(*args)
        yield next(batches)
        if first_attempt:
            first_attempt = False
            raise IOError("transient scan")
        yield from batches

    monkeypatch.setattr(lance_datasource, "_scan_range_span", read)
    assert [
        batch["id"].to_pylist() for batch in source.read_planned_batches(ranges)
    ] == [values[4 * int(r.unit) + r.start : 4 * int(r.unit) + r.stop] for r in ranges]


def test_planned_lance_declines_ambiguous_branch_tag(tmp_path):
    path = str(tmp_path / "tbl.lance")
    main = lance.write_dataset(pa.table({"id": [0, 1]}), path)
    branch = main.create_branch("work")
    branch.update({"id": "id + 10"})
    main.update({"id": "id + 20"})
    main.tags.create("work-snapshot", ("work", branch.version))
    assert main.version == branch.version
    assert main.to_table()["id"].to_pylist() == [20, 21]
    assert branch.to_table()["id"].to_pylist() == [10, 11]
    assert main.tags.list()["work-snapshot"]["branch"] == "work"
    # Even a tag targeting another branch reports the same URI and version
    # as the different main snapshot. Neither is enough to identify that tag.
    tagged = lance.dataset(path, version="work-snapshot")
    assert (tagged.uri, tagged.version) == (main.uri, main.version)
    source = LanceDatasource(path, version="work-snapshot")
    assert not source.supports_read_batch_planning()
    with pytest.raises(ValueError, match="does not support planned read batches"):
        _plan(source, 1, 1)
    assert LanceDatasource(path, version=main.version).supports_read_batch_planning()


def test_planned_lance_relative_uri_uses_driver_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lance.write_dataset(pa.table({"id": [1, 2]}), "relative.lance")
    source = LanceDatasource("relative.lance")
    ranges = _plan(source, 2, 1)
    reader = source.get_read_batch_reader()
    other = tmp_path / "worker"
    other.mkdir()
    monkeypatch.chdir(other)
    lance.write_dataset(pa.table({"id": [3, 4]}), "relative.lance")
    restored = pickle.loads(pickle.dumps(reader))
    assert [r.source for r in ranges] == [str(tmp_path / "relative.lance")]
    assert [table["id"].to_pylist() for table in restored(ranges)] == [[1, 2]]
    # Serializing a warm reader drops native handles and their manifest caches.
    assert pickle.dumps(restored) == pickle.dumps(reader)


def test_planned_lance_empty_projection(fragmented_data):
    path, _, _ = fragmented_data
    source = LanceDatasource(path, columns=[], scanner_options={"batch_size": 1})
    ranges = _plan(source, 3, 1)
    assert [
        (b.num_columns, b.num_rows) for b in source.read_planned_batches(ranges)
    ] == [(0, 3), (0, 1), (0, 3), (0, 1), (0, 1)]


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"filter": "id<3"}, id="predicate"),
        pytest.param({"scan_in_order": False}, id="unordered"),
        pytest.param({"limit": 1}, id="limit"),
        pytest.param({"offset": 3}, id="offset"),
        pytest.param({"order_by": ["id"]}, id="sort"),
        pytest.param({"include_deleted_rows": True}, id="deleted_rows"),
        pytest.param({"unknown_future_option": 1}, id="unrecognized"),
        pytest.param({"columns": {"random_number": "random()"}}, id="computed"),
    ],
)
def test_planned_lance_scan_declines_unknown_row_identity(tmp_path, options):
    path = str(tmp_path / "tbl.lance")
    lance.write_dataset(pa.table({"id": [0, 1, 2]}), path)
    assert not LanceDatasource(
        path, scanner_options=options
    ).supports_read_batch_planning()
    assert not LanceDatasource([path, path]).supports_read_batch_planning()


@pytest.fixture
def fragmented_data(tmp_path):
    path = str(tmp_path / "fragments.lance")
    # The source values deliberately do not encode their physical positions.
    values = [19, 2, 31, 7, 43, 11, 13, 5, 0]
    dataset = lance.write_dataset(
        pa.table({"id": values}), path, max_rows_per_file=4, max_rows_per_group=2
    )
    return path, values, dataset.version


@pytest.mark.parametrize(
    "batch_size,expected",
    [
        pytest.param(
            3, [(0, 0, 3), (0, 3, 4), (1, 0, 3), (1, 3, 4), (2, 0, 1)], id="explicit"
        ),
        pytest.param(None, [(0, 0, 4), (1, 0, 4), (2, 0, 1)], id="estimated"),
    ],
)
def test_planned_lance_udf_batches(
    batch_size,
    expected,
    fragmented_data,
    restore_data_context,
    ray_start_regular_shared,
):
    from ray.data.datasource.planned_batch import (
        get_current_read_batch,
        iter_planned_read_batches,
    )

    path, values, version = fragmented_data
    restore_data_context.enable_planned_map_batches = True

    def describe(table):
        span = get_current_read_batch()
        assert span is not None
        return pa.table(
            {
                "fragment": [int(span.unit)],
                "start": [span.start],
                "stop": [span.stop],
                "version": [int(span.version)],
                "values": [table["id"].to_pylist()],
            }
        )

    expected_rows = [
        {
            "fragment": f,
            "start": a,
            "stop": b,
            "version": version,
            "values": values[4 * f + a : 4 * f + b],
        }
        for f, a, b in expected
    ]
    for parallelism in [1, 5]:
        ds = ray.data.read_lance(path, override_num_blocks=parallelism).map_batches(
            describe, batch_size=batch_size, batch_format="pyarrow"
        )
        discovered = list(iter_planned_read_batches(ds))
        result = ds.take_all()
        assert sorted(
            (int(r.unit), r.start, r.stop, int(r.version)) for r in discovered
        ) == sorted(
            (r["fragment"], r["start"], r["stop"], r["version"]) for r in result
        )
        assert (
            sorted(result, key=lambda row: (row["fragment"], row["start"]))
            == expected_rows
        )


@pytest.mark.parametrize(
    "enabled,scan_options,planned",
    [
        pytest.param(True, {}, True, id="planned"),
        pytest.param(False, {}, False, id="disabled"),
        pytest.param(True, {"scan_in_order": False}, False, id="unordered_scan"),
    ],
)
def test_lance_planned_batch_write_schema(
    enabled,
    scan_options,
    planned,
    fragmented_data,
    restore_data_context,
    ray_start_regular_shared,
    tmp_path,
):
    from ray.data.datasource.planned_batch import get_current_read_batch

    path, values, _ = fragmented_data
    restore_data_context.enable_planned_map_batches = enabled

    def business_output(table):
        assert (get_current_read_batch() is not None) == planned
        return pa.table({"business_id": table["id"]})

    dest = str(tmp_path / "result")
    ray.data.read_lance(path, scanner_options=scan_options).map_batches(
        business_output,
        batch_size=3,
        batch_format="pyarrow",
    ).write_parquet(dest)
    assert sorted(
        ray.data.read_parquet(dest).take_all(), key=lambda row: row["business_id"]
    ) == [{"business_id": value} for value in sorted(values)]


def test_reuse_planned_dataset_with_different_consumer(
    fragmented_data, restore_data_context, ray_start_regular_shared
):
    from ray.data.datasource.planned_batch import (
        get_current_read_batch,
        iter_planned_read_batches,
    )

    path, values, _ = fragmented_data
    restore_data_context.enable_planned_map_batches = True

    def mark(table):
        return pa.table(
            {
                "id": table["id"],
                "planned": [get_current_read_batch() is not None] * len(table),
            }
        )

    ds = ray.data.read_lance(path).map_batches(
        mark, batch_size=3, batch_format="pyarrow"
    )
    assert sorted((r["id"], r["planned"]) for r in ds.take_all()) == [
        (value, True) for value in sorted(values)
    ]
    # The execution has populated the dataset's schema cache. The public DAG
    # must still be reusable when another consumer cannot use planned batches.
    assert ds.schema(fetch_if_missing=False).names == ["id", "planned"]
    batch_map = ds._logical_plan.dag
    assert isinstance(batch_map, MapBatches)
    assert batch_map.batch_size == 3
    (read,) = batch_map.input_dependencies
    assert isinstance(read, Read)
    assert read.planned_map_batch_size is None

    limited_ds = ds.limit(2)
    with pytest.raises(ValueError, match="does not execute source-planned batches"):
        list(iter_planned_read_batches(limited_ds))
    limited = limited_ds.take_all()
    assert len(limited) == 2
    assert all(not row["planned"] and row["id"] in values for row in limited)


def test_fresh_schema_probe_preserves_planned_generator_input(
    fragmented_data, restore_data_context, ray_start_regular_shared
):
    from ray.data.datasource.planned_batch import get_current_read_batch

    path, _, _ = fragmented_data
    restore_data_context.enable_planned_map_batches = True

    def describe(table):
        span = get_current_read_batch()
        assert span is not None
        assert len(table) == span.stop - span.start
        yield pa.table({"offset": [span.start]})

    ds = ray.data.read_lance(path).map_batches(
        describe,
        batch_size=3,
        batch_format="pyarrow",
        compute=ray.data.TaskPoolStrategy(size=100),
    )
    assert ds.schema(fetch_if_missing=False) is None
    assert ds.schema().names == ["offset"]
    assert sorted(row["offset"] for row in ds.take_all()) == [0, 0, 0, 3, 3]


def test_second_map_batches_uses_the_first_maps_output_rows(
    fragmented_data, restore_data_context, ray_start_regular_shared
):
    from ray.data.datasource.planned_batch import get_current_read_batch

    path, values, _ = fragmented_data
    restore_data_context.enable_planned_map_batches = True

    def collapse(table):
        span = get_current_read_batch()
        assert span is not None
        assert len(table) == span.stop - span.start
        return pa.table({"values": [table["id"].to_pylist()]})

    def count_business_rows(table):
        assert get_current_read_batch() is None
        return pa.table(
            {"business_rows": [len(table)], "ranges": [table["values"].to_pylist()]}
        )

    outputs = (
        ray.data.read_lance(path, override_num_blocks=1)
        .map_batches(
            collapse,
            batch_size=3,
            batch_format="pyarrow",
            compute=ray.data.TaskPoolStrategy(size=2),
        )
        .map_batches(count_business_rows, batch_size=2, batch_format="pyarrow")
        .take_all()
    )
    assert sorted(row["business_rows"] for row in outputs) == [1, 2, 2]
    assert sorted(
        v for row in outputs for group in row["ranges"] for v in group
    ) == sorted(values)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
