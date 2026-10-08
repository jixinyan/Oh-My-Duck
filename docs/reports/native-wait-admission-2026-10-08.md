# Native task wait deadlines

`NativeTaskClient.wait(timeout_s=...)` requires a finite positive number before
requesting task status. The recorded voice-task CLI checks the same requirement
before creating outputs or opening model services. Explicit valid deadlines
continue through native task state validation.

Source `6099b17` passed 19 focused checks: eleven invalid API values, three valid
API deadlines and five actual CLI rejection processes. The tests use the actual
client, close its HTTP resources and confirm invalid CLI values create no output
directory. NaN, infinity and nonpositive deadlines fail at their public entry.

The independent wheel, source distribution and installation passed 431
source/resource files, three licenses and 24 CLI calls. Evidence is retained in
`outputs/acceptance/native-wait-admission-20261008-01-install/result.json`.
These changes affect task waiting and voice CLI admission; policy inference,
physics and native ActionGate execution retain their existing implementations.
GPU acceptance and RL remain stopped.
