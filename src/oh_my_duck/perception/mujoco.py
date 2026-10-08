import base64
from io import BytesIO

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from oh_my_duck.perception.rgbd import measure_target


APARTMENT_OBJECTS = {
    "kit_counter": "counter", "kit_fridge": "fridge", "kit_island": "table",
    "kit_stool1": "chair", "kit_stool2": "chair", "liv_sofa": "sofa",
    "liv_table": "table", "liv_tv_stand": "cabinet", "liv_shelf": "shelf",
    "liv_books": "books", "liv_lamp": "lamp", "liv_chair": "chair",
    "liv_plant": "plant", "cor_console": "table", "cor_coat_rack": "rack",
    "cor_shoes": "rack", "bed_base": "bed", "bed_pillow": "pillow",
    "bed_blanket": "blanket", "bed_ns_l": "table", "bed_ns_r": "table",
    "bed_wardrobe": "cabinet", "bed_dresser": "cabinet", "off_desk": "desk",
    "off_chair": "chair", "off_shelf": "shelf", "off_cabinet": "cabinet",
    "off_plant": "plant", "bth_tub": "bathtub", "bth_vanity": "sink",
    "bth_mirror": "mirror", "bth_toilet": "toilet", "bth_rail": "rail",
    "dock": "dock", "obj_0": "block", "obj_1": "block", "obj_2": "block",
    "obj_3": "block", "ball_0": "ball", "ball_1": "ball", "ball_2": "ball",
}


def capture_world_points(renderer):
    if renderer.scene.stereo != mujoco.mjtStereo.mjSTEREO_NONE:
        raise ValueError("Perception requires a monocular head camera")
    camera = mujoco.mjv_averageCamera(*renderer.scene.camera)
    if camera.orthographic:
        raise ValueError("Perception requires the official perspective head camera")
    near, far = float(camera.frustum_near), float(camera.frustum_far)
    bottom, top = float(camera.frustum_bottom), float(camera.frustum_top)
    half_width = float(camera.frustum_width)
    if half_width == 0:
        half_width = (top - bottom) * renderer.width / renderer.height / 2
    if not np.isfinite([near, far, bottom, top, half_width, camera.frustum_center]).all() or not (
        0 < near < far and bottom < top and half_width > 0
    ):
        raise ValueError("Head camera frustum is invalid")
    renderer.enable_depth_rendering()
    try:
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    try:
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    if depth.shape != (renderer.height, renderer.width) or segmentation.shape != (*depth.shape, 2):
        raise ValueError("Depth and segmentation must match the head RGB geometry")
    # RGB、depth 和 segmentation 均使用从上到下的像素中心。
    x = camera.frustum_center + ((np.arange(renderer.width) + 0.5) / renderer.width * 2 - 1) * half_width
    y = top - (np.arange(renderer.height) + 0.5) / renderer.height * (top - bottom)
    forward, up = np.asarray(camera.forward), np.asarray(camera.up)
    right = np.cross(forward, up)
    eye = np.asarray(camera.pos, dtype=float)
    if not np.isfinite([eye, forward, up, right]).all():
        raise FloatingPointError("Head camera pose is nonfinite")
    directions = forward + x[None, :, None] / near * right + y[:, None, None] / near * up
    points = eye + depth[..., None] * directions
    valid = (np.isfinite(depth) & (depth >= near) & (depth < far)
             & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
    points[~valid] = np.nan
    return points.astype(np.float32), segmentation, eye


def inspect_apartment_frame(frame, segmentation, model, robot_root_id, prompt):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 120:
        raise ValueError("Perception prompt must contain 1–120 characters")
    points = np.load(BytesIO(base64.b64decode(frame["points_world_npy_base64"])), allow_pickle=False)
    groups = {}
    for geom_id in np.unique(segmentation[segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, 0]):
        if not 0 <= geom_id < model.ngeom:
            raise ValueError("Segmentation contains an invalid native geom ID")
        if int(model.body_rootid[model.geom_bodyid[geom_id]]) == robot_root_id:
            continue
        name = model.geom(int(geom_id)).name
        for asset, label in APARTMENT_OBJECTS.items():
            if (name == asset or name.startswith(asset + "_")) and prompt in ("objects", label, asset, name):
                groups.setdefault(asset, {"label": label, "ids": []})["ids"].append(int(geom_id))
                break
    image = Image.open(BytesIO(base64.b64decode(frame["rgb_png_base64"]))).convert("RGB")
    draw = ImageDraw.Draw(image)
    targets = []
    for asset, item in groups.items():
        mask = ((segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
                & np.isin(segmentation[..., 0], item["ids"]))
        rows, columns = np.nonzero(mask)
        if len(rows) < 8:
            continue
        box = [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]
        targets.append({"target_id": asset, "label": item["label"], "bbox_xyxy": box,
                        "visible_pixels": len(rows), "detection_source": "simulator_ground_truth",
                        "mask_source": "simulator_ground_truth", "distance_source": "simulator_ground_truth",
                        **measure_target(mask, points, frame["camera_position_m"],
                                         frame["body_position_m"], frame["yaw_rad"])})
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], box[1]), item["label"], fill="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return {"episode_id": frame["episode_id"], "sequence": frame["sequence"],
            "observed_at": frame["observed_at"], "prompt": prompt,
            "detection_source": "simulator_ground_truth", "distance_source": "simulator_ground_truth",
            "targets": sorted(targets, key=lambda target: -target["visible_pixels"])[:16],
            "rgb_png_base64": base64.b64encode(output.getvalue()).decode("ascii")}
