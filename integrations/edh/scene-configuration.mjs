import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';


export async function validateSceneConfiguration(configuration, edhRoot) {
  const { default: Ajv2020 } = createRequire(resolve(edhRoot, 'harness/contracts/package.json'))('ajv/dist/2020.js');
  const schema = JSON.parse(await readFile(new URL(
    '../../src/oh_my_duck/integrations/edh/scene.schema.json', import.meta.url), 'utf8'));
  const validate = new Ajv2020({ strict: true }).compile(schema);
  if (!validate(configuration))
    throw new Error(`MicroDuck scene configuration is invalid: ${JSON.stringify(validate.errors)}`);
}
