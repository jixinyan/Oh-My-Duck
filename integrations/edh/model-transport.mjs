import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { randomUUID } from 'node:crypto';

export function identifyModelClient(undici, baseURL, createParser, auditPath) {
  const origin = new URL(baseURL).origin;
  mkdirSync(dirname(auditPath), { recursive: true });
  const record = (value) => appendFileSync(auditPath,
    `${JSON.stringify({ at: new Date().toISOString(), ...value })}\n`);
  class ModelResponseAudit extends undici.DecoratorHandler {
    constructor(handler, options) {
      super(handler);
      this.requestId = randomUUID();
      this.route = options.path;
      this.decoder = new TextDecoder();
      this.sse = false;
      this.errorChunks = [];
      this.errorBytes = 0;
      this.errorJSON = false;
      this.parser = createParser({
        maxBufferSize: 32 * 1024 * 1024,
        onError: (error) => { throw error; },
        onEvent: (event) => {
          if (!['response.created', 'response.completed', 'response.incomplete',
            'response.failed', 'error'].includes(event.event)) return;
          const data = JSON.parse(event.data);
          const response = data.response ?? data;
          record({ requestId: this.requestId, route: this.route, event: event.event,
            providerRequestId: response.id ?? null, model: response.model ?? null,
            status: response.status ?? null, error: response.error ?? data.error ?? null,
            usage: response.usage ?? null });
        },
      });
    }
    onResponseStart(controller, status, headers, message) {
      this.sse = String(headers['content-type'] ?? '').startsWith('text/event-stream');
      this.errorJSON = status >= 400 &&
        String(headers['content-type'] ?? '').includes('application/json');
      record({ requestId: this.requestId, route: this.route, event: 'http.response', status,
        providerRequestId: headers['x-request-id'] ?? null,
        retryAfter: headers['retry-after'] ?? null });
      return super.onResponseStart(controller, status, headers, message);
    }
    onResponseData(controller, chunk) {
      if (this.sse) this.parser.feed(this.decoder.decode(chunk, { stream: true }));
      if (this.errorJSON) {
        this.errorBytes += chunk.length;
        if (this.errorBytes > 2 * 1024 * 1024)
          throw new Error('Model error response exceeds the audit size limit');
        this.errorChunks.push(Buffer.from(chunk));
      }
      return super.onResponseData(controller, chunk);
    }
    onResponseEnd(controller, trailers) {
      if (this.sse) this.parser.feed(this.decoder.decode());
      if (this.errorJSON) {
        const data = JSON.parse(Buffer.concat(this.errorChunks).toString('utf8'));
        const error = data.error ?? data;
        record({ requestId: this.requestId, route: this.route, event: 'http.error',
          error: { type: error.type ?? null, code: error.code ?? null,
            message: error.message ?? null } });
      }
      return super.onResponseEnd(controller, trailers);
    }
    onResponseError(controller, error) {
      record({ requestId: this.requestId, route: this.route, event: 'transport.error',
        error: { code: error.code ?? null, message: error.message } });
      return super.onResponseError(controller, error);
    }
  }
  const previous = undici.getGlobalDispatcher();
  const dispatcher = new undici.Agent().compose((dispatch) => (options, handler) => {
    if (new URL(options.origin).origin !== origin) return dispatch(options, handler);
    const headers = Array.isArray(options.headers)
      ? undici.util.parseHeaders(options.headers) : { ...options.headers };
    headers['user-agent'] = 'Oh-My-Duck/0.1';
    return dispatch({ ...options, headers }, new ModelResponseAudit(handler, options));
  });
  undici.setGlobalDispatcher(dispatcher);
  return async () => {
    undici.setGlobalDispatcher(previous);
    await dispatcher.close();
  };
}
