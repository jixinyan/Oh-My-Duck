import csv
from dataclasses import dataclass
from io import StringIO
import math


GPU_QUERY = (
    "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
PROCESS_QUERY = (
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader",
)


@dataclass(frozen=True)
class GpuDevice:
    index: int
    uuid: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float

    @property
    def memory_available_mib(self) -> float:
        return self.memory_total_mib - self.memory_used_mib


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    memory_display: str


def gpu_devices(report: str) -> tuple[GpuDevice, ...]:
    devices = []
    for row in csv.reader(StringIO(report), skipinitialspace=True, strict=True):
        if len(row) != 5:
            raise ValueError("GPU 状态记录需要五个 CSV 字段")
        index, uuid, utilization, used, total = (value.strip() for value in row)
        device = GpuDevice(int(index), uuid, float(utilization), float(used), float(total))
        if (device.index < 0 or not device.uuid or
                not all(math.isfinite(value) for value in
                        (device.utilization_percent, device.memory_used_mib, device.memory_total_mib)) or
                not 0 <= device.utilization_percent <= 100 or
                not 0 <= device.memory_used_mib <= device.memory_total_mib or device.memory_total_mib <= 0):
            raise ValueError("GPU 状态记录包含无效设备、使用比例或显存数值")
        devices.append(device)
    if not devices or len({device.index for device in devices}) != len(devices) or (
            len({device.uuid for device in devices}) != len(devices)):
        raise ValueError("GPU 状态记录需要完整且唯一的设备身份")
    return tuple(devices)


def compute_processes(report: str) -> tuple[ComputeProcess, ...]:
    processes = []
    for row in csv.reader(StringIO(report), skipinitialspace=True, strict=True):
        if len(row) != 4:
            raise ValueError("GPU 进程记录需要四个 CSV 字段")
        uuid, pid, name, memory = (value.strip() for value in row)
        process = ComputeProcess(uuid, int(pid), name, memory)
        if not process.gpu_uuid or process.pid <= 0 or not process.process_name or not process.memory_display:
            raise ValueError("GPU 进程记录缺少有效设备、进程身份或显存记录")
        processes.append(process)
    return tuple(processes)


def require_exclusive_device(devices: tuple[GpuDevice, ...], processes: tuple[ComputeProcess, ...],
                             index: int, minimum_available_mib: float = 16384) -> GpuDevice:
    if type(index) is not int or index < 0:
        raise ValueError("GPU 设备编号必须为非负整数")
    if (isinstance(minimum_available_mib, bool) or not math.isfinite(minimum_available_mib) or
            minimum_available_mib < 0):
        raise ValueError("GPU 可用显存要求必须为有限的非负数值")
    selected = [device for device in devices if device.index == index]
    if len(selected) != 1:
        raise ValueError("GPU 状态记录缺少请求设备的唯一身份")
    device = selected[0]
    occupied = sorted({process.pid for process in processes if process.gpu_uuid == device.uuid})
    if occupied:
        raise RuntimeError(f"GPU {index} 已有 compute PID {occupied}，需要独占设备")
    if device.memory_available_mib < minimum_available_mib:
        raise RuntimeError(f"GPU {index} 的可用显存不足 {minimum_available_mib} MiB")
    return device
