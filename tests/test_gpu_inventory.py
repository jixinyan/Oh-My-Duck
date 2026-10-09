import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest

from oh_my_duck.infrastructure.gpu_inventory import (
    GPU_QUERY, PROCESS_QUERY, compute_processes, gpu_devices, require_exclusive_device,
)


RECORD = Path(__file__).parent / 'records/gpu-inventory-20261009'


def recorded_inventory():
    manifest = json.loads((RECORD / 'capture.json').read_text())
    devices = (RECORD / 'devices.csv').read_text()
    processes = (RECORD / 'processes.csv').read_text()
    for name, report in (('devices.csv', devices), ('processes.csv', processes)):
        assert hashlib.sha256(report.encode()).hexdigest() == manifest['queries'][name]['sha256']
    assert manifest['queries']['devices.csv']['command'] == list(GPU_QUERY)
    assert manifest['queries']['processes.csv']['command'] == list(PROCESS_QUERY)
    assert manifest['gpu_work_started'] is False and manifest['cuda_initialized'] is False
    return manifest, devices, processes


def csv_report(rows):
    output = StringIO()
    csv.writer(output).writerows(rows)
    return output.getvalue()


def test_actual_inventory_identity_and_counts():
    manifest, device_report, process_report = recorded_inventory()
    devices, processes = gpu_devices(device_report), compute_processes(process_report)
    assert len(devices) == manifest['device_count'] == 8
    assert len(processes) == manifest['compute_rows']
    assert sorted({process.pid for process in processes}) == manifest['unique_compute_pids']
    assert all(device.memory_available_mib == device.memory_total_mib - device.memory_used_mib for device in devices)


def test_zero_utilization_does_not_admit_an_occupied_device():
    _, device_report, process_report = recorded_inventory()
    devices, processes = gpu_devices(device_report), compute_processes(process_report)
    occupied = next(device for device in devices if device.utilization_percent == 0 and
                    any(process.gpu_uuid == device.uuid for process in processes))
    with pytest.raises(RuntimeError, match='compute PID') as rejected:
        require_exclusive_device(devices, processes, occupied.index)
    pids = sorted({process.pid for process in processes if process.gpu_uuid == occupied.uuid})
    assert str(pids) in str(rejected.value)


def test_recorded_unoccupied_device_preserves_identity_and_headroom():
    _, device_report, process_report = recorded_inventory()
    devices, processes = gpu_devices(device_report), compute_processes(process_report)
    available = next(device for device in devices if device.memory_available_mib >= 16384 and
                     not any(process.gpu_uuid == device.uuid for process in processes))
    assert require_exclusive_device(devices, processes, available.index) is available
    with pytest.raises(RuntimeError, match='可用显存'):
        require_exclusive_device(devices, processes, available.index, available.memory_available_mib + 1)


@pytest.mark.parametrize('field,value', [(2, 'nan'), (2, 'inf'), (2, '-inf'), (2, '-1'), (2, '101'),
                                        (3, '-1'), (3, 'nan'), (4, '0'), (4, 'inf'), (0, '-1')])
def test_invalid_device_values_are_rejected(field, value):
    _, report, _ = recorded_inventory()
    rows = list(csv.reader(StringIO(report), skipinitialspace=True))
    rows[0][field] = value
    with pytest.raises(ValueError):
        gpu_devices(csv_report(rows))


def test_duplicate_device_identity_is_rejected():
    _, report, _ = recorded_inventory()
    rows = list(csv.reader(StringIO(report), skipinitialspace=True))
    with pytest.raises(ValueError, match='唯一'):
        gpu_devices(csv_report([*rows, rows[0]]))
    rows[1][1] = rows[0][1]
    with pytest.raises(ValueError, match='唯一'):
        gpu_devices(csv_report(rows))


def test_incomplete_device_record_is_rejected():
    _, report, _ = recorded_inventory()
    rows = list(csv.reader(StringIO(report), skipinitialspace=True))
    with pytest.raises(ValueError, match='五个'):
        gpu_devices(csv_report([rows[0][:-1]]))
    with pytest.raises(ValueError, match='完整'):
        gpu_devices('')


@pytest.mark.parametrize('field,value', [(0, ''), (1, '0'), (1, '-1'), (1, 'nan'), (2, ''), (3, '')])
def test_invalid_process_values_are_rejected(field, value):
    _, _, report = recorded_inventory()
    rows = list(csv.reader(StringIO(report), skipinitialspace=True))
    rows[0][field] = value
    with pytest.raises(ValueError):
        compute_processes(csv_report(rows))


def test_incomplete_process_record_is_rejected():
    _, _, report = recorded_inventory()
    rows = list(csv.reader(StringIO(report), skipinitialspace=True))
    with pytest.raises(ValueError, match='四个'):
        compute_processes(csv_report([rows[0][:-1]]))


def test_actual_process_duplicates_preserve_original_rows():
    manifest, _, report = recorded_inventory()
    processes = compute_processes(report)
    assert len(processes) == manifest['compute_rows'] > len(manifest['unique_compute_pids'])
    assert len(set(processes)) < len(processes)


def test_empty_process_selection_and_missing_device():
    _, device_report, process_report = recorded_inventory()
    devices, processes = gpu_devices(device_report), compute_processes(process_report)
    device = next(device for device in devices if not any(p.gpu_uuid == device.uuid for p in processes))
    selected_report = csv_report([row for row in csv.reader(StringIO(process_report), skipinitialspace=True)
                                 if row[0] == device.uuid])
    assert compute_processes(selected_report) == ()
    with pytest.raises(ValueError, match='唯一身份'):
        require_exclusive_device(tuple(d for d in devices if d.index != device.index), processes, device.index)


@pytest.mark.parametrize('index', [-1, True, 2.0])
def test_invalid_requested_device_is_rejected(index):
    _, device_report, process_report = recorded_inventory()
    with pytest.raises(ValueError, match='设备编号'):
        require_exclusive_device(gpu_devices(device_report), compute_processes(process_report), index)


@pytest.mark.parametrize('minimum', [-1, True, float('nan'), float('inf')])
def test_invalid_memory_requirement_is_rejected(minimum):
    _, device_report, process_report = recorded_inventory()
    with pytest.raises(ValueError, match='显存要求'):
        require_exclusive_device(gpu_devices(device_report), compute_processes(process_report), 2, minimum)
