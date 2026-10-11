import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 22 || (major === 22 && minor < 19))
  throw new Error('原生 Harness 要求 Node.js 22.19.0 或更新版本');
if (process.argv[2] === '--node-only') {
  process.stdout.write(`${JSON.stringify({ node: process.versions.node })}\n`);
} else {
  const source = resolve(process.argv[2]);
  const [{ ContractValidator }, models, server] = await Promise.all([
    import(resolve(source, 'harness/contracts/src/index.ts')),
    import(resolve(source, 'harness/agent-runtime/models/src/index.ts')),
    import(resolve(source, 'apps/server/src/index.ts')),
  ]);
  const exports = {
    ContractValidator,
    OpenAICompatibleAdapter: models.OpenAICompatibleAdapter,
    OpenAIResponsesAdapter: models.OpenAIResponsesAdapter,
    startServer: server.startServer,
    createNativeWorkerEnvironment: server.createNativeWorkerEnvironment,
  };
  for (const [name, value] of Object.entries(exports))
    if (typeof value !== 'function') throw new Error(`原生 SDK 缺少导出: ${name}`);
  new ContractValidator(JSON.parse(await readFile(resolve(source,
    'harness/contracts/schema/physical.schema.json'), 'utf8')));
  process.stdout.write(`${JSON.stringify({ node: process.versions.node,
    exports: Object.keys(exports), physicalSchemaLoaded: true })}\n`);
}
