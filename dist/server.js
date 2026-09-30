// Process entry point: argument parsing + static server lifecycle.
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { createRequestHandler, parseArgs } from './src/http-app.js';

const rootDir = dirname(fileURLToPath(import.meta.url));
const { host, port } = parseArgs(process.argv);

const server = createServer(createRequestHandler(rootDir));
server.on('error', (err) => {
  console.error(err.message);
  process.exit(1);
});
server.listen(port, host, () => {
  const address = server.address();
  const realPort = typeof address === 'object' && address ? address.port : port;
  const shownHost = host === '0.0.0.0' ? 'localhost' : host;
  console.log(`注意力掩码工作台运行于 http://${shownHost}:${realPort}`);
  console.log(`(绑定 ${host}:${port}，实际端口 ${realPort})`);
});
