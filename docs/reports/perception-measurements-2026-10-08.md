# Original RGBD measurement validation

The model service and client verify target geometry against the original RGBD
frame. Source admission requires RGB PNG, matching floating-point world-point
geometry, finite camera/body pose, nonnegative sequence and explicit frame identity.
Missing depth pixels contain three NaN coordinates.

YOLO measurements use the rounded bounding-box region. SAM returns a source-sized
binary grayscale PNG mask and a matching tight box. Camera distance, 10%–90%
interval, median world position, horizontal body distance, body bearing and valid
depth count are recomputed. Numeric comparison uses absolute tolerance `1e-6`.
Insufficient depth supplies its status and pixel count.

`frames.py` owns source decoding, box pixels and mask PNG. `rgbd.py` retains the
original measurement calculations. `validation.py` owns response and source
consistency; `client.py` and `service.py` apply both checks. Native tool text omits
the mask's base64 payload. Source locations are in the
[perception source map](../../src/oh_my_duck/perception/README.md).

## Actual CPU execution

Fixed source `ef770314da146d45b68fdad0bacc6e9183d57344` ran the actual locked
Ultralytics `8.4.170` / Torch `2.10.0+cu130` service on `jd_B300`, using `--device cpu`
and empty `CUDA_VISIBLE_DEVICES`. YOLO26s SHA256 is
`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.
Explicit prediction devices follow the
[official inference arguments](https://docs.ultralytics.com/modes/predict/#inference-arguments).

The five original frames come from
`outputs/acceptance/cpu-scene-perception-20261008-03`: corridor, corridor after
75 actual policy controls, kitchen, living room and office. Their original bytes
are preserved; this campaign performs new model inference on those saved frames.

| Check | Result |
| --- | --- |
| Actual HTTP model requests | Five original frames, five model targets |
| Original-RGBD geometry | Every target's measurements independently recomputed |
| Invalid measurement/source admission | 90 rejected alterations |
| PNG masks | Original target pixels preserved during encode/decode; wrong dimensions rejected |
| Device | YOLO parameters remain on CPU; CUDA uninitialized before and after inference |
| Resource release | Service and two validation processes exited zero; PIDs absent and service TCP connection refused |
| Independent installation | 436 source/resource files, three licenses, 27 CLI calls and the actual saved perception audit outside the checkout |
| Existing interface regression | 72 tests and 28 subtests passed for lazy imports, complete tool Schemas and native scene configuration |

Original artifacts and source hashes are retained in:

- `outputs/acceptance/perception-measurements-20261008-01`.
- `outputs/acceptance/perception-measurements-independent-20261008-01`.
- `outputs/acceptance/perception-measurements-driver-20261008-01`.
- `outputs/acceptance/perception-measurements-release-20261008-01/result.json`.
- `outputs/acceptance/perception-measurements-20261008-01-install/result.json`.
- `outputs/acceptance/perception-measurements-20261008-01-install/independent-perception.json`.

The retained two-view Newton YOLO record also passes its 48 identity/geometry
admission checks at the current source. It has its original saved-data scope.

## Public command and remaining acceptance

`omd validate perception --frames DIRECTORY --endpoint URL --output NEW_DIRECTORY`
calls the actual service and saves original frame bytes, responses, model/device
metadata and SHA256. `--capture DIRECTORY` audits those saved measurements.
See [service and command examples](../perception-navigation.md).

Current-source SAM mask transport and inference require the declared GPU campaign.
Independent labelled recognition and distance accuracy, object tracking, image-driven
navigation and actual robot calibration remain pending. GPU execution and RL stay
stopped. All original physical control, task, policy and stopping requirements remain
in effect.
