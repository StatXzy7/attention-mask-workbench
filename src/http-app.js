// Request handler factory, separated from process startup so tests can import
// parsing/handler logic without opening a listening socket.
import { readFile } from 'node:fs/promises';
import { extname, normalize, resolve, sep } from 'node:path';

export const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

export function parseArgs(argv, env = process.env) {
  const args = { host: env.HOST || '127.0.0.1', port: env.PORT };
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    let key;
    let value;
    if (token.includes('=')) {
      [key, value] = token.slice(2).split('=');
    } else {
      key = token.slice(2);
      value = argv[i + 1];
      i += 1;
    }
    if (key === 'host' || key === 'port') args[key] = value;
  }
  let port;
  if (args.port === undefined) {
    port = 8080;
  } else {
    port = Number(args.port);
    if (!Number.isInteger(port) || port < 0 || port > 65535) {
      throw new Error(`Invalid --port: ${args.port}`);
    }
  }
  return { host: args.host, port };
}

export function createRequestHandler(rootDir) {
  return async function handler(req, res) {
    try {
      const url = new URL(req.url, 'http://localhost');
      const pathname = url.pathname === '/' ? '/web/index.html' : decodeURIComponent(url.pathname);
      const filePath = resolve(rootDir, '.' + sep + normalize(pathname));
      const insideRoot =
        filePath === rootDir || filePath.startsWith(rootDir + sep);
      if (!insideRoot) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }
      if (filePath === rootDir) {
        res.writeHead(404);
        res.end('Not Found');
        return;
      }
      const body = await readFile(filePath);
      res.writeHead(200, {
        'Content-Type': MIME[extname(filePath)] || 'application/octet-stream',
      });
      res.end(body);
    } catch (err) {
      if (err && err.code === 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('404 Not Found');
      } else {
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end(`500 ${err.message}`);
      }
    }
  };
}
