import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from oh_my_duck.rl.artifacts.identity import verify_export_source
from oh_my_duck.infrastructure.provenance import validate_resume


class ArtifactIdentityTests(unittest.TestCase):
    def test_checkpoint_and_policy_must_match_export_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);checkpoint=p/'model_4.pt';onnx=p/'policy.onnx'
            checkpoint.write_bytes(b'checkpoint');onnx.write_bytes(b'policy')
            report={'source_checkpoint':str(checkpoint), 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'policy_sha256':hashlib.sha256(onnx.read_bytes()).hexdigest()}
            onnx.with_suffix('.provenance.json').write_text(json.dumps(report))
            self.assertEqual(verify_export_source(onnx,checkpoint),report)
            onnx.write_bytes(b'other policy')
            with self.assertRaises(ValueError):verify_export_source(onnx,checkpoint)

    def test_resume_rejects_another_task_backend_or_framework(self):
        run={'task':'walking','backend':'mujoco','framework':'sb3'}
        validate_resume(run,**run)
        for key in run:
            expected={**run,key:'other'}
            with self.assertRaises(ValueError):validate_resume(run,**expected)
