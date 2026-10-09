# 客户端音频动态库检查

`omd doctor --runtime voice-client` 核查锁定的 Python 依赖，加载实际
PortAudio 与 libsndfile，并记录库版本。动态库不可用时在加载位置报错。
`voice-session` 使用调用者明确选择的输入和输出设备。
Linux 安装说明位于 [语音交互](../voice-interaction.md)。

## 已验证范围

同一份 `cli/doctor.py` 的 SHA256 为
`97d39cb80e845b06412940ffdca0010eafae849ff24b72698ff0beeef39c3cdd`。

- macOS 实际加载 PortAudio 19.7 与 libsndfile 1.2.2，检查通过。
  记录为 `outputs/acceptance/mac-voice-library-readiness-20261009-01.json`。
- Ubuntu 22.04 实际加载 PortAudio 19.6 与 libsndfile 1.2.2，全部动态库依赖
  能够解析；没有动态库搜索路径时，检查直接拒绝该环境。
  `voice-session --help` 和 `voice-task --help` 均返回实际命令说明。
  记录为 `outputs/acceptance/linux-voice-audio-20261009-01/linux-result.json`。
- Ubuntu 使用发行版的 PortAudio、ALSA、JACK 及对应依赖。
  包版本、SHA256、动态链接结果和命令输出保存在服务器
  `outputs/acceptance/linux-voice-audio-20261009-01/` 及
  `outputs/acceptance/linux-voice-audio-development-20261009-01/`。
- 源码 `c6c6139d04304d9ac01da646f5ffe923356ba1ef` 的 Linux wheel、sdist
  与独立安装检查通过 438 个源码和资源文件、三份 license、27 项 CLI 调用。
  记录位于服务器
  `outputs/acceptance/remote-voice-client-20261009-02/package-audio-audit.json`。

检查使用 HTTPX 0.28.1、NumPy 2.5.3、sounddevice 0.5.6 与 soundfile 0.14.0。
CUDA 保持未使用。这个范围覆盖库加载、依赖检查与命令入口。
麦克风录音、扬声器播放和 Microduck 音频设备需要对应的设备执行记录。
