'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const WEB_DIR = path.join(__dirname, '..', 'web');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
};

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, body) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404 Not Found');
      return;
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' });
    res.end(body);
  });
}

function createServer() {
  return http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    let pathname = decodeURIComponent(url.pathname);
    if (pathname === '/') pathname = '/index.html';
    const baseDir = pathname.startsWith('/src/')
      ? path.join(WEB_DIR, '..', 'src')
      : WEB_DIR;
    const rel = pathname.startsWith('/src/') ? pathname.slice('/src/'.length - 1) : pathname;
    const filePath = path.normalize(path.join(baseDir, rel));
    if (!filePath.startsWith(baseDir + path.sep) && filePath !== baseDir) {
      res.writeHead(403);
      res.end('403 Forbidden');
      return;
    }
    fs.stat(filePath, (err, stat) => {
      if (err || !stat.isFile()) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('404 Not Found');
        return;
      }
      sendFile(res, filePath);
    });
  });
}

function parseArgs(argv) {
  const opts = { host: '127.0.0.1', port: 5173 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    let key = null;
    let value = null;
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq >= 0) {
        key = a.slice(2, eq);
        value = a.slice(eq + 1);
      } else {
        key = a.slice(2);
        value = argv[i + 1];
        i += 1;
      }
    }
    if (key === 'host' && value !== undefined) opts.host = value;
    if (key === 'port' && value !== undefined) {
      const p = Number(value);
      if (!Number.isInteger(p) || p < 0 || p > 65535) {
        throw new Error(`无效端口: ${value}`);
      }
      opts.port = p;
    }
  }
  return opts;
}

function start(options = {}) {
  const host = options.host || '127.0.0.1';
  const port = options.port === undefined ? 5173 : options.port;
  const server = createServer();
  return new Promise((resolve) => {
    server.listen(port, host, () => {
      const addr = server.address();
      resolve({
        server,
        host,
        port: addr.port,
        address: `http://${host}:${addr.port}/`,
      });
    });
  });
}

if (require.main === module) {
  const opts = parseArgs(process.argv.slice(2));
  start(opts).then(({ address }) => {
    console.log(`稀疏注意力掩码工作台已启动: ${address}`);
  }).catch((err) => {
    console.error(err.message);
    process.exit(1);
  });
}

module.exports = { createServer, parseArgs, start };
