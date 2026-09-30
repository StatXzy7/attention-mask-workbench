'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { createServer, parseArgs, start } = require('../src/server.js');

test('parseArgs 支持 --host/--port 及等号形式, 端口 0 合法', () => {
  assert.deepEqual(parseArgs(['--host', '0.0.0.0', '--port', '8080']), { host: '0.0.0.0', port: 8080 });
  assert.deepEqual(parseArgs(['--host=::1', '--port=0']), { host: '::1', port: 0 });
  assert.throws(() => parseArgs(['--port', '70000']), /无效端口/);
});

test('动态端口 0 启动并输出可用地址, 页面与共享模块可经 HTTP 获取后自行结束', async () => {
  const { server, host, port, address } = await start({ host: '127.0.0.1', port: 0 });
  try {
    assert.ok(port > 0);
    assert.equal(address, `http://127.0.0.1:${port}/`);

    const home = await fetch(address);
    assert.equal(home.status, 200);
    const html = await home.text();
    assert.match(html, /本地稀疏注意力掩码工作台/);
    assert.match(html, /src="\/app\.js"/);

    const appJs = await fetch(`${address}app.js`);
    assert.equal(appJs.status, 200);
    assert.match(await appJs.text(), /renderHeadSvg/);

    // 浏览器复用的状态/核心模块通过 /src/ 暴露
    const storeJs = await fetch(`${address}src/store.js`);
    assert.equal(storeJs.status, 200);
    assert.match(await storeJs.text(), /createStore/);
    const coreJs = await fetch(`${address}src/core.js`);
    assert.equal(coreJs.status, 200);
    assert.match(await coreJs.text(), /computeWorkbench/);

    const css = await fetch(`${address}styles.css`);
    assert.equal(css.status, 200);

    const missing = await fetch(`${address}nope.txt`);
    assert.equal(missing.status, 404);

    // 路径穿越被拒
    const traversal = await fetch(`${address}..%2f..%2fpackage.json`);
    assert.ok(traversal.status === 403 || traversal.status === 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('createServer 返回可正常关闭的服务器实例', async () => {
  const server = createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  await new Promise((resolve) => server.close(resolve));
});
