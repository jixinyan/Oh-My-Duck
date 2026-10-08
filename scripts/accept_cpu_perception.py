import argparse
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image
import torch

from oh_my_duck.perception.mujoco import APARTMENT_OBJECTS
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend


def validate_case(backend, output, name, pose, control_steps=0):
    backend.reset_episode(42, {"kind": "room", "room": "corridor", "hold_ticks": 5}, pose)
    for index in range(control_steps):
        inference = backend.infer_policy()
        backend.apply_policy_action(inference['action'], request_id=f'{name}:{index}',
                                    expected_sequence=inference['sequence'])
    if control_steps:
        assert backend._stopped_samples >= 5
    before = [backend.data.qpos.copy(), backend.data.qvel.copy(), backend.data.ctrl.copy()]
    counters = (backend._sequence, float(backend.data.time), backend._stopped_samples, backend._goal_held_ticks)
    frame, segmentation = backend._perception_capture()
    points = np.load(BytesIO(base64.b64decode(frame['points_world_npy_base64'])), allow_pickle=False)
    rgb = np.asarray(Image.open(BytesIO(base64.b64decode(frame['rgb_png_base64']))))
    assert points.shape == (240, 320, 3) and rgb.shape == (240, 320, 3)
    assert rgb.std() > 0 and frame['sequence'] == control_steps
    assert np.allclose(frame['camera_position_m'], backend.data.cam_xpos[backend._camera_site_id], atol=1e-7)
    checked = []
    for row in range(5, 235, 13):
        for column in range(5, 315, 17):
            geom, kind = segmentation[row, column]
            if kind != mujoco.mjtObj.mjOBJ_GEOM or not np.isfinite(points[row, column]).all():
                continue
            if backend.model.body_rootid[backend.model.geom_bodyid[geom]] == backend._robot_root_id:
                continue
            if not all(np.array_equal(segmentation[row, column], segmentation[r, c])
                       for r, c in ((row - 2, column), (row + 2, column),
                                    (row, column - 2), (row, column + 2))):
                continue
            camera_id = backend._camera_site_id
            assert backend.model.cam_sensorsize[camera_id, 0] == 0
            focal = 240 / (2 * math.tan(math.radians(float(backend.model.cam_fovy[camera_id])) / 2))
            rotation = backend.data.cam_xmat[camera_id].reshape(3, 3)
            ray = rotation @ np.array([(column + 0.5 - 160) / focal, (120 - row - 0.5) / focal, -1])
            # 射线从实际 near plane 开始，使用模型 fovy 独立计算像素方向。
            near = float(backend.model.vis.map.znear * backend.model.stat.extent)
            origin = backend.data.cam_xpos[camera_id] + ray * near
            ray /= np.linalg.norm(ray)
            distance = mujoco.mju_rayGeom(backend.data.geom_xpos[geom], backend.data.geom_xmat[geom],
                                          backend.model.geom_size[geom], origin, ray,
                                          int(backend.model.geom_type[geom]))
            assert distance >= 0, (name, row, column, geom)
            point = origin + ray * distance
            error = float(np.linalg.norm(points[row, column] - point))
            assert error < 0.002, (name, row, column, error)
            checked.append({'row': row, 'column': column, 'geom_id': int(geom),
                            'selected_world_m': point.tolist(), 'error_m': error})
    assert len(checked) >= 50
    invalid = segmentation[..., 1] != mujoco.mjtObj.mjOBJ_GEOM
    assert np.isnan(points[invalid]).all()
    inspection = backend.inspect_scene('objects')
    assert inspection['sequence'] == frame['sequence'] and inspection['episode_id'] == frame['episode_id']
    assert inspection['targets'], name
    for target in inspection['targets']:
        asset = target['target_id']
        assert target['label'] == APARTMENT_OBJECTS[asset]
        ids = [i for i in range(backend.model.ngeom)
               if backend.model.geom(i).name == asset or backend.model.geom(i).name.startswith(asset + '_')]
        mask = np.isin(segmentation[..., 0], ids) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        rows, columns = np.nonzero(mask)
        assert target['bbox_xyxy'] == [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]
        assert target['visible_pixels'] == len(rows)
        selected = points[mask & np.isfinite(points).all(axis=-1)]
        assert len(selected) >= 8 and target['valid_depth_pixels'] == len(selected)
        center = np.median(selected, axis=0)
        distances = np.linalg.norm(selected - frame['camera_position_m'], axis=-1)
        assert abs(target['distance_m'] - float(np.median(distances))) < 1e-6
        assert np.allclose(target['surface_position_world_m'], center, atol=1e-7)
        dx, dy = center[:2] - frame['body_position_m'][:2]
        bearing = math.atan2(dy, dx) - frame['yaw_rad']
        assert abs(target['bearing_deg'] - math.degrees(math.atan2(math.sin(bearing), math.cos(bearing)))) < 1e-5
        assert target['detection_source'] == target['mask_source'] == target['distance_source'] == 'simulator_ground_truth'
    assert backend.inspect_scene('unrepresented_target')['targets'] == []
    public_frame = backend.perception_frame()
    assert public_frame['episode_id'] == frame['episode_id'] and public_frame['sequence'] == frame['sequence']
    next_rgb = np.asarray(Image.open(BytesIO(base64.b64decode(backend.observe_control()['measurements']['rgb_png_base64']))))
    assert np.array_equal(rgb, next_rgb)
    for original, current in zip(before, [backend.data.qpos, backend.data.qvel, backend.data.ctrl]):
        assert np.array_equal(original, current)
    assert counters == (backend._sequence, float(backend.data.time), backend._stopped_samples, backend._goal_held_ticks)
    (output / f'{name}-frame.json').write_text(json.dumps(frame) + '\n')
    (output / f'{name}-inspection.json').write_text(json.dumps(inspection) + '\n')
    np.savez_compressed(output / f'{name}-geometry.npz', points_world_m=points, segmentation=segmentation)
    return {'case': name, 'pose': pose, 'control_steps': control_steps, 'native_ray_pixels': len(checked),
            'maximum_world_point_error_m': max(item['error_m'] for item in checked),
            'targets': inspection['targets'], 'pixels': checked, 'physical_state_unchanged': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    backend = CpuMujocoBamBackend(robot_id='cpu-perception-acceptance', catalog_dir=args.catalog)
    try:
        from OpenGL import GL

        cases = [('corridor', 0.0, 0.0, 0.0, 0), ('office', 2.0, -0.4, 0.0, 0),
                 ('kitchen', -2.8, 0.8, 1.2, 0), ('living_room', -1.6, -0.5, math.pi, 0),
                 ('corridor_after_policy', 0.0, 0.0, 0.0, 75)]
        results = [validate_case(backend, args.output, name, {'x_m': x, 'y_m': y, 'yaw_rad': yaw}, steps)
                   for name, x, y, yaw, steps in cases]
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        assert 'llvmpipe' in renderer.lower(), renderer
        assert not torch.cuda.is_initialized()
    finally:
        backend.close()
    root = Path(__file__).resolve().parents[1]
    result = {'passed': True, 'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'mujoco_version': mujoco.__version__, 'renderer': renderer, 'cases': results,
              'cuda_initialized': torch.cuda.is_initialized(), 'resources_closed': True,
              'artifacts': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in args.output.iterdir() if path.is_file()}}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'cases': len(results),
                      'native_ray_pixels': sum(item['native_ray_pixels'] for item in results),
                      'maximum_world_point_error_m': max(item['maximum_world_point_error_m'] for item in results),
                      'renderer': renderer, 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
