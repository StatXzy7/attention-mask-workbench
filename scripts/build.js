'use strict';

// 无打包步骤的项目: build 负责加载核心模块并对全部交付文件做语法/可达性检查
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { computeWorkbench } = require('../src/core.js');

const root = path.join(__dirname, '..');
const files = [
  'src/core.js',
  'src/store.js',
  'src/server.js',
  'scripts/build.js',
  'web/app.js',
];

let failed = false;
for (const rel of files) {
  const res = spawnSync(process.execPath, ['--check', path.join(root, rel)], { encoding: 'utf8' });
  if (res.status !== 0) {
    failed = true;
    console.error(`语法检查失败: ${rel}\n${res.stderr}`);
  } else {
    console.log(`语法检查通过: ${rel}`);
  }
}

for (const rel of ['web/index.html', 'web/styles.css']) {
  if (!fs.existsSync(path.join(root, rel))) {
    failed = true;
    console.error(`缺少静态资源: ${rel}`);
  } else {
    console.log(`静态资源存在: ${rel}`);
  }
}

// 冒烟计算: 零预算 + 因果必须给出空掩码和零向量
const q = [[[1, 0], [0, 1]]];
const k = [[[1, 0], [0, 1]]];
const v = [[[1, 2], [3, 4]]];
const smoke = computeWorkbench(q, k, v, { numBlocks: 2, budget: 0, causal: true });
if (smoke.metrics.totalRetained !== 0) {
  failed = true;
  console.error('冒烟计算失败: 零预算应无保留位置');
}
if (JSON.stringify(smoke.sparseOut) !== JSON.stringify([[[0, 0], [0, 0]]])) {
  failed = true;
  console.error('冒烟计算失败: 零预算稀疏输出应为零向量');
}

if (failed) {
  console.error('build 失败');
  process.exit(1);
}
console.log('build 成功: 所有检查通过');
