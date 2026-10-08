import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';


export function perceptionSources(configuration) {
  if (!Object.hasOwn(configuration ?? {}, 'perception_endpoint')) return ['simulator_ground_truth'];
  const endpoint = configuration.perception_endpoint;
  if (typeof endpoint !== 'string' || endpoint.trim() !== endpoint ||
      !/^http:\/\/127\.0\.0\.1:[0-9]{1,5}\/?$/.test(endpoint))
    throw new Error('Perception endpoint must be an explicit local service or SSH tunnel');
  const address = new URL(endpoint);
  const port = Number(address.port || 80);
  if (!Number.isSafeInteger(port) || port < 1)
    throw new Error('Perception endpoint must use a valid TCP port');
  return ['simulator_ground_truth', 'models'];
}


export async function validateSceneConfiguration(configuration, edhRoot) {
  const { default: Ajv2020 } = createRequire(resolve(edhRoot, 'harness/contracts/package.json'))('ajv/dist/2020.js');
  const schema = JSON.parse(await readFile(new URL(
    '../../src/oh_my_duck/integrations/edh/scene.schema.json', import.meta.url), 'utf8'));
  const validate = new Ajv2020({ strict: true }).compile(schema);
  if (!validate(configuration))
    throw new Error(`MicroDuck scene configuration is invalid: ${JSON.stringify(validate.errors)}`);
  perceptionSources(configuration);
}
