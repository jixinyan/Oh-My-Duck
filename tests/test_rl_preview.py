import json
from pathlib import Path
import tempfile
import unittest
from oh_my_duck.rl.experiments.preview import checkpoint_source


class PreviewCheckpointSelection(unittest.TestCase):
    def test_partial_sb3_save_cannot_replace_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); campaign=root/'campaign'; run=campaign/'stand'; run.mkdir(parents=True)
            output=root/'training'; complete=output/'checkpoints/step_000100'; complete.mkdir(parents=True)
            for name in ('model.zip','vecnormalize.pkl','run.json'):
                (complete/name).write_text('{}')
            partial=output/'checkpoints/.step_000200.partial'; partial.mkdir(); (partial/'model.zip').write_text('partial')
            incomplete=output/'checkpoints/step_000300'; incomplete.mkdir(); (incomplete/'model.zip').write_text('incomplete')
            (run/'result.json').write_text(json.dumps({'stages':{'full':{'command':['train','--output',str(output)]}}}))
            spec={'id':'stand','framework':'sb3'}
            self.assertEqual(checkpoint_source(spec,campaign,root),(complete,complete/'model.zip'))

    def test_recovered_run_can_preview_original_checkpoint_before_new_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'source'
            spec={'id':'walking','framework':'rsl-rl','resume':{'run':str(run),'checkpoint':'model_250.pt'}}
            self.assertEqual(checkpoint_source(spec,root/'new',root),(run,run/'model_250.pt'))
