# Copyright 2026 Pollen Robotics. Apache-2.0.
# Modified by Oh My Duck: extracted runner into the owned RL package.
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}



    def save(self, path: str, infos=None):
        if not hasattr(self.env.unwrapped.scene["robot"], "reference_model"):
            return super().save(path, infos)
        # Keep native checkpoint/optimizer/curriculum persistence and native ONNX
        # export. Only the metadata model view differs for Newton's DOF actuators.
        from mjlab.rl.runner import MjlabOnPolicyRunner
        from mjlab.rl.exporter_utils import attach_metadata_to_onnx
        from oh_my_duck.rl.artifacts.metadata import policy_metadata
        MjlabOnPolicyRunner.save(self, path, infos)
        directory, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(directory), filename)
            attach_metadata_to_onnx(str(onnx_path), policy_metadata(self.env.unwrapped, str(path)))
        except Exception as error:
            # The durable native checkpoint remains valid if a preview fails.
            print(f"[WARN] ONNX export failed (training continues): {error}")
