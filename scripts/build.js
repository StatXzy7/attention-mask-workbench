// Build: syntax-check every JS source, then stage the runnable app into dist/.
import { mkdir, rm, cp, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { readdir } from 'node:fs/promises';

const ROOT = new URL('..', import.meta.url).pathname;

async function listJs(dir) {
  const out = [];
  async function walk(current) {
    const entries = await readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name === 'node_modules' || entry.name === '.git' || entry.name === 'dist') {
        continue;
      }
      const full = join(current, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (entry.name.endsWith('.js')) {
        out.push(full);
      }
    }
  }
  await walk(dir);
  return out.sort();
}

const files = await listJs(ROOT);
let failed = 0;
for (const file of files) {
  const res = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  if (res.status !== 0) {
    failed += 1;
    console.error(`✗ syntax error in ${file}\n${res.stderr || res.stdout}`);
  } else {
    console.log(`✓ ${file.replace(ROOT, '')}`);
  }
}
if (failed > 0) {
  console.error(`Build failed: ${failed} file(s) with syntax errors.`);
  process.exit(1);
}

const dist = join(ROOT, 'dist');
await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
await cp(join(ROOT, 'web'), join(dist, 'web'), { recursive: true });
await cp(join(ROOT, 'src'), join(dist, 'src'), { recursive: true });
await cp(join(ROOT, 'server.js'), join(dist, 'server.js'));
await cp(join(ROOT, 'package.json'), join(dist, 'package.json'));

const manifest = {
  name: 'attention-mask-workbench',
  builtAt: new Date().toISOString(),
  node: process.version,
  files: files.map((f) => f.replace(ROOT, '')),
  thirdPartyDependencies: 0,
};
await writeFile(join(dist, 'build-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`Build succeeded: ${files.length} JS files checked, dist/ staged.`);
