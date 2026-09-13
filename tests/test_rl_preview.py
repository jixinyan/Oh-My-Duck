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


def test_training_preview_stops_with_its_worker_even_if_training_raises(tmp_path):
    from unittest.mock import patch, MagicMock
    import pytest
    from oh_my_duck.rl.experiments.preview import training_preview
    child = MagicMock(pid=123)
    child.wait.return_value = 0
    output = tmp_path / 'run'
    output.mkdir()
    with patch('oh_my_duck.rl.experiments.preview.subprocess.Popen', return_value=child) as spawn:
        with pytest.raises(RuntimeError, match='training failure'):
            with training_preview(tmp_path, output, {'id':'walker'}, '3', 1000) as record:
                assert record['gpu'] == '3'
                raise RuntimeError('training failure')
    assert (output / 'preview-training-finished').exists()
    assert record['status'] == 'completed'
    child.wait.assert_called_once()
    command = spawn.call_args.args[0]
    assert command[command.index('--gpu')+1] == '3'
    assert command[command.index('--minimum-updates')+1] == '1000'
    assert not spawn.call_args.kwargs.get('start_new_session',False)
