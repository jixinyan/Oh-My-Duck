import argparse
import importlib
import sys


COMMANDS = {
    "metric": ("metric.campaign", "执行真实 policy tool 的运动指标验收"),
    "metric-case": ("metric.case", "执行单个原生物理会话的运动序列"),
    "metric-admission": ("harness.metric_admission", "检查准备中的动作请求、实际执行和明确修改命令"),
    "execution-retry": ("harness.execution_retry", "检查原生执行重试、连续物理状态与目标判定"),
    "metric-audit": ("metric.verify", "复核运动样本、停止状态、相机与源码来源"),
    "policy-audit": ("metric.policy", "使用官方 ONNX 重新计算记录中的 policy action"),
    "policy-registry": ("harness.registry", "核验登记的 policy 包、来源、命令通道与 CPU 推断"),
    "policy-packages": ("harness.package_execution", "执行登记的长期及限定时长 policy 并检查原生控制"),
    "motion-guard": ("harness.motion_guard", "检查真实运动暂停、恢复与终止"),
    "pose": ("harness.pose", "执行官方 CPU 头部与身体姿态命令验证"),
    "pose-audit": ("harness.pose_records", "复核姿态命令、物理响应、相机与原生动作记录"),
    "replay": ("harness.replay", "检查原生 agent、tool、sensor 与 Verifier 记录"),
    "navigation": ("harness.navigation", "执行声明场景的完整原生导航验收"),
    "navigation-audit": ("harness.navigation_replay", "复核导航轨迹、waypoint 与正式判定"),
    "release": ("release.campaign", "执行远程 metadata 检查或已分配 GPU 的验收"),
    "release-worker": ("release.worker", "检查 worker 的依赖、场景与来源"),
    "release-plan": ("release.plans", "检查声明的全部阶段、配置和指令"),
    "model-assets": ("release.assets", "使用 CPU 核验 USD 碰撞、官方材质和 Newton 导入"),
    "package-audit": ("release.package", "检查 wheel、源码包和独立安装的 CLI"),
}


def main():
    parser = argparse.ArgumentParser(description="Microduck 原生执行与记录验收")
    parser.add_argument("operation", choices=COMMANDS, help="; ".join(
        f"{name}: {description}" for name, (_, description) in COMMANDS.items()))
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return 0
    arguments = parser.parse_args(sys.argv[1:2])
    module = importlib.import_module("oh_my_duck.validation." + COMMANDS[arguments.operation][0])
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    return module.main()
