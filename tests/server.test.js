import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequestHandler, parseArgs, MIME } from '../src/http-app.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

test('parseArgs honors --host/--port, equals forms, and dynamic port 0', () => {
  assert.deepEqual(parseArgs(['node', 'server.js'], {}), { host: '127.0.0.1', port: 8080 });
  assert.deepEqual(parseArgs(['node', 'server.js', '--host', '0.0.0.0', '--port', '3000'], {}), {
    host: '0.0.0.0',
    port: 3000,
  });
  assert.deepEqual(parseArgs(['node', 'server.js', '--host=127.0.0.1', '--port=0'], {}), {
    host: '127.0.0.1',
    port: 0,
  });
  assert.deepEqual(parseArgs(['node', 'server.js'], { PORT: '4100', HOST: '0.0.0.0' }), {
    host: '0.0.0.0',
    port: 4100,
  });
  assert.throws(() => parseArgs(['node', 'server.js', '--port', '70000'], {}), /Invalid --port/);
  assert.throws(() => parseArgs(['node', 'server.js', '--port', 'abc'], {}), /Invalid --port/);
  assert.throws(() => parseArgs(['node', 'server.js', '--port', '-1'], {}), /Invalid --port/);
});

function listenEphemeral(handler) {
  return new Promise((resolve, reject) => {
    const server = createServer(handler);
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

test('request handler serves index/assets/modules and returns 404 for missing files', async () => {
  const server = await listenEphemeral(createRequestHandler(ROOT));
  after(() => server.close());
  const port = server.address().port;

  const html = await fetch(`http://127.0.0.1:${port}/`);
  assert.equal(html.status, 200);
  assert.equal(html.headers.get('content-type'), MIME['.html']);
  assert.match(await html.text(), /稀疏注意力掩码工作台/);

  const appJs = await fetch(`http://127.0.0.1:${port}/web/app.js`);
  assert.equal(appJs.status, 200);
  assert.equal(appJs.headers.get('content-type'), MIME['.js']);

  const css = await fetch(`http://127.0.0.1:${port}/web/style.css`);
  assert.equal(css.status, 200);
  assert.equal(css.headers.get('content-type'), MIME['.css']);

  const core = await fetch(`http://127.0.0.1:${port}/src/core.js`);
  assert.equal(core.status, 200);
  assert.match(await core.text(), /analyzeModel/);

  const missing = await fetch(`http://127.0.0.1:${port}/nope.txt`);
  assert.equal(missing.status, 404);
});

test('request handler blocks path traversal outside the root', async () => {
  const server = await listenEphemeral(createRequestHandler(ROOT));
  after(() => server.close());
  const port = server.address().port;
  const res = await fetch(`http://127.0.0.1:${port}/../../etc/passwd`);
  assert.ok(res.status === 403 || res.status === 404);
});

function spawnServer(args) {
  const child = spawn(process.execPath, ['server.js', ...args], { cwd: ROOT });
  child.stdout.setEncoding('utf8');
  let output = '';
  const ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('startup timed out: ' + output)), 6000);
    child.stdout.on('data', (chunk) => {
      output += chunk;
      // Wait for both address lines so the "actual port" line is captured.
      const match = output.match(/http:\/\/[^:\s]+:(\d+)/);
      if (match && output.includes('实际端口')) {
        clearTimeout(timer);
        resolve({ port: Number(match[1]), output });
      }
    });
    child.on('error', reject);
    child.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`server exited early (${code}): ${output}`));
    });
  });
  return { child, ready };
}

test('real process: --port 0 prints the bound address and self-terminates on SIGTERM', async () => {
  const { child, ready } = spawnServer(['--host', '127.0.0.1', '--port', '0']);
  const { port, output } = await ready;
  assert.ok(port > 0);
  assert.match(output, /实际端口/);

  const res = await fetch(`http://127.0.0.1:${port}/web/render.js`);
  assert.equal(res.status, 200);

  child.kill('SIGTERM');
  const [code] = await once(child, 'exit');
  assert.ok(code === 0 || code === null);
});
