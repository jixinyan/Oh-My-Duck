# Perception source map

| Responsibility | Implementation |
| --- | --- |
| Active observation and target interfaces | `base.py` |
| RGBD admission, PNG masks and YOLO box pixels | `frames.py` |
| Target surface distance, interval, world position and body bearing | `rgbd.py` |
| Response identity and original-RGBD measurement verification | `validation.py` |
| Explicit local model service or SSH-tunnel client | `client.py` |
| YOLO26 and optional SAM3.1 inference | `service.py`, `sam31.py` |
| Actual CPU camera depth/segmentation and named apartment targets | `mujoco.py` |
| Actual Newton ray data and scene targets | `../robotics/backends/isaac_official.py` |
| Native `inspect_scene` and combined `observe` tools | `../integrations/edh/session.py` |
| Model calls, retained source frames and independent measurement audit | `../validation/perception.py` / `omd validate perception` |

The client decodes the original RGB PNG, floating-point world coordinates and
camera/body pose before making a model request. A missing depth pixel has three
NaN coordinates. Image geometry, finite pose, frame identity and explicit distance
source are required.

`validate_response` checks identity, model/image SHA256, target fields and
annotation geometry. `validate_measurements` reconstructs every measured target
from the original frame. Both the service and client use these checks. YOLO uses
the rounded bounding-box region; SAM returns its source-sized binary mask as
`mask_png_base64`. Mask PNG contains only zero and 255, with grayscale mode `L`.
SAM boxes must enclose exactly the returned mask. Measurement comparison uses an
absolute tolerance of `1e-6` for the existing floating-point calculations.

The distance calculation reports median camera-to-surface distance, the 10%–90%
interval, median world position, body-to-surface horizontal distance and body
bearing. Fewer than eight valid depth pixels produce `insufficient_valid_depth`
without geometric measurements. Source consistency and arithmetic verification
have their own scope; recognition accuracy requires independently labelled data.

YOLO can run explicitly on `cpu` or `cuda:N`. The pinned SAM predictor requires
`cuda:0` within one explicitly selected physical GPU. CPU selection does not
initialize CUDA. Runtime environment, model acquisition, scene configuration and
public command examples are in
[perception and navigation](../../../docs/perception-navigation.md).
