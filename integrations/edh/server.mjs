import { randomBytes, randomUUID } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createConnection, createServer } from 'node:net';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { identifyModelClient } from './model-transport.mjs';
import { createRequire } from 'node:module';

const omdRoot = fileURLToPath(new URL('../../', import.meta.url));
const edhRoot = resolve(process.env.OMD_EDH_SOURCE ?? '');
const pinnedRevision = '8a5e685b22d032207f53db20454f0992a4ad60fd';
const python = process.env.OMD_CPU_PYTHON;
const catalogDir = process.env.OMD_POLICY_DIR;
const dataDirectory = process.env.OMD_DATA_DIRECTORY;
const baseURL = process.env.EDH_MODEL_BASE_URL;
const model = process.env.EDH_MODEL;
const modelAPI = process.env.EDH_MODEL_API ?? 'chat-completions';
const key = process.env.EDH_MODEL_API_KEY;
const reasoningEffort = process.env.EDH_REASONING_EFFORT;
const remoteWorker = process.env.OMD_REMOTE_WORKER
  ? JSON.parse(process.env.OMD_REMOTE_WORKER) : null;
if (reasoningEffort && !['low', 'medium', 'high', 'xhigh', 'max'].includes(reasoningEffort))
  throw new Error('Model reasoning effort is invalid');
if (!python || !catalogDir || !dataDirectory || !baseURL || !model)
  throw new Error('Simulation worker, policy catalog, data directory, and model endpoint are required');
if (!['chat-completions', 'responses'].includes(modelAPI))
  throw new Error('Model API is invalid');
const [{ ContractValidator }, { OpenAICompatibleAdapter, OpenAIResponsesAdapter }, serverModule] = await Promise.all([
  import(resolve(edhRoot, 'harness/contracts/src/index.ts')),
  import(resolve(edhRoot, 'harness/agent-runtime/models/src/index.ts')),
  import(resolve(edhRoot, 'apps/server/src/index.ts')),
]);
const { startServer, createNativeWorkerEnvironment } = serverModule;
const ModelAdapter = modelAPI === 'responses' ? OpenAIResponsesAdapter : OpenAICompatibleAdapter;
const { createParser } = createRequire(resolve(edhRoot,
  'harness/agent-runtime/models/package.json'))('eventsource-parser');
const closeModelTransport = identifyModelClient(
  await import(resolve(edhRoot, 'node_modules/undici/index.js')), baseURL, createParser,
  resolve(dataDirectory, 'model-transport.jsonl'));
if (process.env.OMD_CHECK_MODEL === '1') {
  const adapter = new ModelAdapter({ baseURL,
    models: [{ id: model, inputModalities: ['text', 'image'], contextWindow: 32768, maxTokens: 4096 }],
    ...(key ? { apiKey: () => key } : {}), timeoutMs: 180_000 });
  const info = await adapter.resolveModel('configured-vlm', model);
  if (reasoningEffort && !info.reasoning?.efforts.some((effort) => effort.id === reasoningEffort))
    throw new Error('Configured adapter does not expose the requested reasoning effort');
  let text = '';
  let finish;
  for await (const chunk of adapter.stream({ provider: 'configured-vlm', model,
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Reply OK.' }] }],
    maxTokens: 4096, ...(reasoningEffort ? { reasoningEffort } : {}),
    signal: new AbortController().signal })) {
    if (chunk.type === 'text-delta') text += chunk.text;
    if (chunk.type === 'finish') finish = chunk.reason;
  }
  if (!text.trim() || finish?.kind !== 'stop') throw new Error('Model probe did not complete with text');
  await closeModelTransport();
  process.stdout.write(`${JSON.stringify({ model, modelAPI, reasoningEffort, finish, text })}\n`);
  process.exit(0);
}
const validator = new ContractValidator(JSON.parse(await readFile(
  resolve(edhRoot, 'harness/contracts/schema/physical.schema.json'), 'utf8')));
const tempDirectory = resolve(omdRoot, '.cache/tmp');
await mkdir(tempDirectory, { recursive: true });
await mkdir(dataDirectory, { recursive: true });
const runSockets = new Map();
const sceneConfiguration = process.env.OMD_SCENE_CONFIGURATION
  ? JSON.parse(process.env.OMD_SCENE_CONFIGURATION) : null;
if (sceneConfiguration !== null) {
  if (sceneConfiguration.backend !== 'isaac-newton' ||
      typeof sceneConfiguration.scene_id !== 'string' ||
      !/^[a-z0-9][a-z0-9._-]*$/.test(sceneConfiguration.scene_id) ||
      typeof sceneConfiguration.usd_path !== 'string' ||
      typeof sceneConfiguration.provenance_path !== 'string' ||
      typeof sceneConfiguration.task_instruction !== 'string' ||
      !sceneConfiguration.task_instruction.trim() ||
      sceneConfiguration.goal?.kind !== 'point' ||
      !Array.isArray(sceneConfiguration.goal.target_xy_m) ||
      sceneConfiguration.goal.target_xy_m.length !== 2 ||
      !sceneConfiguration.goal.target_xy_m.every(Number.isFinite) ||
      !Number.isFinite(sceneConfiguration.goal.distance_m) ||
      sceneConfiguration.goal.distance_m <= 0 ||
      !Number.isSafeInteger(sceneConfiguration.goal.hold_ticks) ||
      sceneConfiguration.goal.hold_ticks < 1 ||
      !Number.isSafeInteger(sceneConfiguration.budget?.max_control_steps) ||
      sceneConfiguration.budget.max_control_steps < 1 ||
      !Number.isFinite(sceneConfiguration.budget?.max_wall_time_s) ||
      sceneConfiguration.budget.max_wall_time_s <= 0)
    throw new Error('Isaac scene configuration is incomplete or invalid');
}
const nativeTaskId = sceneConfiguration?.scene_id ?? 'official-apartment-office';
const goal = sceneConfiguration?.goal ?? { kind: 'room', room: 'office', hold_ticks: 5 };
const taskInstruction = sceneConfiguration?.task_instruction ?? 'Navigate the official MicroDuck apartment from the corridor to the office and remain upright in the office for five admitted control steps.';
const environmentLabel = sceneConfiguration
  ? `Isaac Lab Newton/BAM: ${nativeTaskId}` : 'CPU MuJoCo/BAM official 8 × 6 m apartment';
const check = { check_id: 'goal_reached', check: 'native_goal_reached', args: [] };
const catalog = {
  revision: `microduck-${nativeTaskId}-${pinnedRevision.slice(0, 12)}`,
  tasks: {
    [sceneConfiguration ? `navigate-${nativeTaskId}` : 'navigate-office']: {
      label: sceneConfiguration ? `Navigate ${nativeTaskId}` : 'Navigate to the office',
      instruction: taskInstruction,
      goal: {
        id: `${nativeTaskId}-reached`,
        configuration: JSON.stringify(goal),
        successContract: {
          id: `microduck-${nativeTaskId}-native`, version: '1', all: [check],
          source: { kind: 'benchmark', reference: `${nativeTaskId}-native-gt` },
        },
        entities: { robot: 'microduck', destination: sceneConfiguration ? nativeTaskId : 'office' },
        capabilities: ['policy-navigation', 'head-rgb', 'tof', 'imu', 'joint-state', 'odometry'],
        taskSemantics: ['Navigate using admitted policy actions', 'Remain upright at the destination'],
        budget: sceneConfiguration?.budget ?? { max_control_steps: 4000, max_wall_time_s: 1800 },
      },
    },
  },
};

async function control(runId, operation, arguments_, signal) {
  const endpoint = runSockets.get(runId);
  if (!endpoint) throw new Error('MicroDuck task has no active native worker');
  signal.throwIfAborted();
  const socket = createConnection({ host: '127.0.0.1', port: endpoint.port });
  const abort = () => socket.destroy(signal.reason);
  signal.addEventListener('abort', abort, { once: true });
  try {
    await new Promise((accept, reject) => {
      socket.once('connect', accept);
      socket.once('error', reject);
    });
    socket.write(`${JSON.stringify({ run_task_id: runId, control_secret: endpoint.secret, operation,
      request_id: randomUUID(), arguments: arguments_ })}\n`);
    const lines = createInterface({ input: socket, crlfDelay: Infinity });
    for await (const line of lines) {
      const response = JSON.parse(line);
      if (response.error) throw new Error(`${response.error.type}: ${response.error.message}`);
      if (!Object.hasOwn(response, 'result')) throw new Error('MicroDuck tool response has no result');
      return response.result;
    }
    throw new Error('MicroDuck tool connection closed without a response');
  } finally {
    signal.removeEventListener('abort', abort);
    socket.destroy();
  }
}

async function availableControlPort() {
  const server = createServer();
  await new Promise((accept, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', accept);
  });
  const port = server.address().port;
  await new Promise((accept, reject) => server.close((error) => error ? reject(error) : accept()));
  return port;
}

function shellQuote(value) {
  if (typeof value !== 'string' || !value.length || value.includes('\0'))
    throw new Error('Remote worker command contains an invalid argument');
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function workerTransport(controlPort) {
  if (remoteWorker === null) return {
    command: [python, '-m', 'oh_my_duck.integrations.edh_native'],
    env: { PYTHONPATH: `${resolve(omdRoot, 'src')}:${resolve(edhRoot,
      'harness/physical-runtime/src')}`, TMPDIR: tempDirectory },
    schemaPath: resolve(edhRoot, 'harness/contracts/schema/physical.schema.json'),
  };
  for (const field of ['root', 'python', 'edh_source', 'policy_dir'])
    if (typeof remoteWorker[field] !== 'string' || !remoteWorker[field].startsWith('/'))
      throw new Error(`Remote worker ${field} must be an absolute path`);
  if (typeof remoteWorker.host !== 'string' || !remoteWorker.host.length ||
      remoteWorker.host.startsWith('-') || /\s/.test(remoteWorker.host))
    throw new Error('Remote worker SSH host is invalid');
  const remoteTemp = `${remoteWorker.root}/.cache/tmp`;
  const assignments = [
    `PYTHONPATH=${shellQuote(`${remoteWorker.root}/src:${remoteWorker.edh_source}/harness/physical-runtime/src`)}`,
    `TMPDIR=${shellQuote(remoteTemp)}`,
  ];
  if (remoteWorker.cuda_device !== null) {
    if (!Number.isSafeInteger(remoteWorker.cuda_device) || remoteWorker.cuda_device < 0)
      throw new Error('Remote worker CUDA device is invalid');
    assignments.push(`CUDA_VISIBLE_DEVICES=${remoteWorker.cuda_device}`);
  }
  const command = `cd ${shellQuote(remoteWorker.root)} && exec env ${assignments.join(' ')} ` +
    `${shellQuote(remoteWorker.python)} -u -m oh_my_duck.integrations.edh_native 3>&1 1>&2`;
  return {
    command: ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
      '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
      '-L', `127.0.0.1:${controlPort}:127.0.0.1:${controlPort}`, remoteWorker.host, command],
    transportFd: 1,
    env: {},
    schemaPath: `${remoteWorker.edh_source}/harness/contracts/schema/physical.schema.json`,
  };
}

function tool(operation, properties, required, services) {
  return (assignment) => ({
    name: `microduck__${operation}`,
    description: {
      policy_catalog: 'List the verified official MicroDuck policies, their kind and command encoding.',
      scene_info: 'Read the current scene public geometry and navigation information.',
      select_policy: 'Select an official policy at a confirmed stopped execution boundary.',
      transition_policy: 'After a complete episodic policy reaches its native terminal boundary, explicitly select the next policy while preserving physical pose, velocity and action history. This transition does not confirm physical rest. Use a standing policy for recovery and measure the result.',
      finish_policy: 'End the current policy at a confirmed paused boundary; the native independent Verifier then checks the goal.',
      set_command: 'Set a bounded policy command at a confirmed paused boundary. Each command runs for 5–100 actual control steps. After proximity, contact, or stall, read fresh ToF and change twist before renewed motion; zero twist remains available for stopping.',
      read_sensor: 'Read a current physical MicroDuck RGB, ToF, IMU, joint, or odometry sensor.',
      inspect_scene: 'Inspect current head RGB for a named object and return visible targets, bounding boxes, surface distance in meters and bearing in degrees. Select models for YOLO26/SAM service inference, or simulator_ground_truth for explicit native shape masks and ray-hit distances. The result labels detection, mask and distance sources. Requires a confirmed paused execution. Positive bearing means left. Reobserve after movement; targets are tied to one episode and sequence.',
      task_progress: 'Read current physical position, velocity, contact evidence, bounded-command status, and native motion-pause reason.',
      walk: 'Prepare the official locomotion policy (alpha_walking for standard feet, roller for the roller model) to move a signed distance in meters along the current heading. Positive moves forward; negative moves backward. Requires a confirmed paused execution and five measured stopped samples. Call execution.resume afterward. Native odometry controls completion and braking; read task_progress.metric_motion after the pause.',
      rotate: 'Prepare the official locomotion policy (alpha_walking for standard feet, roller for the roller model) to turn by a signed angle in degrees. This maneuver includes translation; the measured translation_xy_m is reported. Positive is counterclockwise around world +Z; negative is clockwise. Requires a confirmed paused execution and five measured stopped samples. Call execution.resume afterward. Accumulated measured yaw controls completion and braking; read task_progress.metric_motion after the pause.',
    }[operation],
    parameters: { type: 'object', properties, required, additionalProperties: false },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [
        { type: 'text', text: JSON.stringify(value) },
        ...(value.image_ref ? [{ type: 'image', attachment: value.image_ref }] : []),
      ],
    },
    timeoutMs: 120_000,
    execute: async (args, exec) => {
      const runId = assignment.brief.task_scope.task_id;
      const workerOperation = {
        policy_catalog: 'catalog', scene_info: 'scene', task_progress: 'progress',
      }[operation] ?? operation;
      const result = await control(runId, workerOperation, args, exec.signal);
      if (operation !== 'inspect_scene' && (operation !== 'read_sensor' || args.sensor !== 'head_rgb')) return result;
      const encoded = operation === 'inspect_scene' ? result.rgb_png_base64 : result.measurements.rgb_png_base64;
      const data = Buffer.from(encoded, 'base64');
      if (!data.length || data.toString('base64') !== encoded)
        throw new Error('MicroDuck RGB transport is invalid');
      const [imageRef] = await services.images.saveImages([
        { data, mediaType: 'image/png', name: 'microduck-head-rgb.png' },
      ]);
      if (!/^sha256:[a-f0-9]{64}$/.test(imageRef.attachmentId))
        throw new Error('Native tool image identity is invalid');
      const imageDirectory = resolve(dataDirectory, 'tool-images');
      await mkdir(imageDirectory, { recursive: true });
      await writeFile(resolve(imageDirectory, `${imageRef.attachmentId.slice(7)}.png`), data);
      if (operation === 'inspect_scene') delete result.rgb_png_base64;
      else delete result.measurements.rgb_png_base64;
      return { ...result, image_ref: imageRef };
    },
  });
}

const port = Number(process.env.OMD_EDH_PORT ?? 4318);
if (!Number.isSafeInteger(port) || port < 1 || port > 65535)
  throw new Error('OMD_EDH_PORT must be a valid TCP port');
const server = await startServer({
  root: edhRoot,
  port,
  dataDirectory,
  deployment: ({ images }) => ({
    id: 'microduck-live',
    version: `edh-${pinnedRevision}-official-policy-1-${modelAPI}`,
    source: 'simulation',
    description: `${environmentLabel} with native EDH execution and verification`,
    teamFile: resolve(omdRoot, 'integrations/edh/team.yaml'),
    roleRoot: resolve(omdRoot, 'integrations/edh'),
    defaultModel: 'brain',
    models: { brain: { provider: 'configured-vlm', model,
      ...(reasoningEffort ? { reasoningEffort } : {}) } },
    adapters: [{ providers: ['configured-vlm'], adapter: new ModelAdapter({
      baseURL, models: [{ id: model, inputModalities: ['text', 'image'],
        contextWindow: 32768, maxTokens: 4096 }],
      ...(key ? { apiKey: () => key } : {}),
      timeoutMs: 180_000,
      resolveImage: (ref, signal) => images.readImageRequest(ref,
        { maxPixels: 1024 * 1024, maxBytes: 2 * 1024 * 1024 }, signal),
    }) }],
    contextManagement: {
      compaction: { thresholdRatio: 0.7, retainRatio: 0.15,
        headroomTokens: 4096, maxTokens: 8192 },
      visualHistory: { maxImages: 12 },
    },
    assignmentLifetimeMs: 3_600_000,
    tasks: {},
    additionalTools: {
      'microduck.policy_catalog': tool('policy_catalog', {}, [], { images }),
      'microduck.scene_info': tool('scene_info', {}, [], { images }),
      'microduck.select_policy': tool('select_policy', {
        policy_name: { type: 'string' },
      }, ['policy_name'], { images }),
      'microduck.transition_policy': tool('transition_policy', {
        policy_name: { type: 'string' },
      }, ['policy_name'], { images }),
      'microduck.finish_policy': tool('finish_policy', {
        execution_id: { type: 'string' },
        generation: { type: 'integer', minimum: 0 },
        boundary_id: { type: 'string' },
      }, ['execution_id', 'generation', 'boundary_id'], { images }),
      'microduck.set_command': tool('set_command', {
        command: { type: 'object', properties: {
          twist: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          head: { type: 'array', items: { type: 'number' }, minItems: 4, maxItems: 4 },
          body: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
          posture: { type: 'string', enum: ['sit', 'stand'] },
        }, additionalProperties: false },
        max_control_steps: { type: 'integer', minimum: 5, maximum: 100 },
      }, ['command'], { images }),
      'microduck.read_sensor': tool('read_sensor', {
        sensor: { type: 'string', enum: ['head_rgb', 'tof', 'imu', 'joint_state', 'odometry'] },
      }, ['sensor'], { images }),
      'microduck.task_progress': tool('task_progress', {}, [], { images }),
      'microduck.inspect_scene': tool('inspect_scene', {
        prompt: { type: 'string', minLength: 1, maxLength: 120 },
        source: { type: 'string', enum: ['models', 'simulator_ground_truth'] },
      }, ['prompt', 'source'], { images }),
      'microduck.walk': tool('walk', {
        distance_m: { type: 'number', minimum: -10, maximum: 10 },
        speed_m_s: { type: 'number', minimum: 0.1, maximum: 0.4 },
      }, ['distance_m'], { images }),
      'microduck.rotate': tool('rotate', {
        angle_deg: { type: 'number', minimum: -360, maximum: 360 },
        angular_speed_deg_s: { type: 'number', minimum: 10, maximum: 55 },
      }, ['angle_deg'], { images }),
    },
    launchProfiles: {
      [nativeTaskId]: {
        source: 'simulation',
        label: sceneConfiguration ? `MicroDuck: ${nativeTaskId}` : 'Official MicroDuck apartment: office',
        environment: environmentLabel,
        embodiment: 'MicroDuck XL330 M6',
        executionMode: 'policy',
        policy: 'official-microduck-onnx',
        checkpoint: 'pollen-robotics/microduck-policies@1b56c396825c052a4e26e95cf2b8d8298af9e9b4',
        defaultModel: 'brain',
        tasks: [],
        taskSource: 'environment',
        createEnvironment: async ({ signal, services }) => {
          signal.throwIfAborted();
          const controlPort = await availableControlPort();
          const controlSecret = randomBytes(32).toString('hex');
          const worker = {
            provider: 'microduck',
            ...workerTransport(controlPort),
            cwd: omdRoot,
            nativeTaskId,
            sceneConfiguration: {
              ...sceneConfiguration,
              backend: sceneConfiguration?.backend ?? 'cpu-mujoco-bam',
              native_task_id: nativeTaskId,
              catalog_dir: catalogDir,
              seed: Number(process.env.OMD_SCENE_SEED ?? 20260929),
              goal,
              spawn_pose: sceneConfiguration?.spawn_pose ?? { x_m: 0, y_m: 0, yaw_rad: 0 },
              task_instruction: taskInstruction,
              policy_revision: '1b56c396825c052a4e26e95cf2b8d8298af9e9b4',
              control_port: controlPort,
              control_secret: controlSecret,
            },
            policyId: 'official-microduck-onnx',
            policyUri: 'microduck-native://self-hosted',
            executionMode: 'policy',
            policyMaxActionsPerInference: 1,
            monitorEveryActions: 1,
            observationTtlS: 30,
            deviceTimeoutS: 120,
            policyTimeoutS: 30,
            catalog,
          };
          let native;
          try {
            native = await createNativeWorkerEnvironment(worker, services, validator);
          } catch (error) {
            console.error(error);
            throw error;
          }
          return {
            describeTasks: native.describeTasks,
            async createTaskBackend(taskId, options) {
              const backend = await native.createTaskBackend(taskId, options);
              runSockets.set(options.runId, { port: controlPort, secret: controlSecret });
              const close = backend.close.bind(backend);
              const stop = backend.stop.bind(backend);
              let latestStatus;
              const observeStatus = backend.subscribe(({ status }) => {
                latestStatus = structuredClone(status);
              });
              backend.stop = async () => {
                try {
                  await stop();
                  if (latestStatus && !(latestStatus.state === 'ended' && latestStatus.device_confirmed)) {
                    await new Promise((resolveStop, rejectStop) => {
                      const timer = setTimeout(() => {
                        unsubscribeStop();
                        rejectStop(new Error('Native execution termination was not confirmed.'));
                      }, 60_000);
                      const unsubscribeStop = backend.subscribe(({ status }) => {
                        if (status.state === 'ended' && status.device_confirmed) {
                          clearTimeout(timer);
                          unsubscribeStop();
                          resolveStop();
                        }
                      });
                    });
                  }
                } catch (error) {
                  console.error(error);
                  throw error;
                }
              };
              backend.close = async () => {
                try {
                  await close();
                } finally {
                  observeStatus();
                  runSockets.delete(options.runId);
                }
              };
              return backend;
            },
            async close() {
              for (const [runId, endpoint] of runSockets)
                if (endpoint.port === controlPort) runSockets.delete(runId);
              await native.close();
            },
          };
        },
      },
    },
  }),
});
process.stdout.write(`${server.url}\n`);
for (const signal of ['SIGINT', 'SIGTERM'])
  process.once(signal, () => {
    void server.close().then(closeModelTransport).then(() => process.exit(0), (error) => {
      process.stderr.write(`${String(error)}\n`);
      process.exit(1);
    });
  });
