// Start a password-protected local server and a temporary Cloudflare Tunnel.

import crypto from 'node:crypto';
import { spawn } from 'node:child_process';

const port = Number(process.env.PORT || 3000);
const password = process.env.TAKHI_SHARE_PASSWORD || crypto.randomBytes(18).toString('base64url');
const server = spawn(process.execPath, ['server.js'], {
  cwd: new URL('..', import.meta.url),
  env: { ...process.env, PORT: String(port), TAKHI_HOST: '127.0.0.1',
    TAKHI_SHARE_PASSWORD: password, TAKHI_SHARE_MODE: '1' },
  stdio: ['ignore', 'pipe', 'pipe']
});

let tunnel;
let ready = false;
let stopping = false;
let serverOutput = '';

function stop() {
  if (stopping) return;
  stopping = true;
  tunnel?.kill('SIGTERM');
  server.kill('SIGTERM');
}

for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { stop(); process.exit(0); });
server.on('error', error => { console.error(`Не удалось запустить сервер: ${error.message}`); stop(); process.exitCode = 1; });
server.on('exit', code => {
  if (!stopping) { console.error(`Сервер остановился (${code}).`); stop(); process.exitCode = 1; }
});
server.stderr.on('data', chunk => process.stderr.write(chunk));
server.stdout.on('data', chunk => {
  if (ready) return;
  serverOutput = (serverOutput + chunk.toString()).slice(-1000);
  if (!serverOutput.includes('Takhi Fieldbook:')) return;
  ready = true;
  tunnel = spawn('cloudflared', ['tunnel', '--no-autoupdate', '--url', `http://127.0.0.1:${port}`], {
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let output = '';
  let publicUrl;
  let connected = false;
  const connectTimeout = setTimeout(() => {
    if (!connected) { console.error('Туннель не подключился за 90 секунд.'); stop(); process.exitCode = 1; }
  }, 90000);
  const observe = chunk => {
    output = (output + chunk.toString()).slice(-10000);
    const match = output.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/);
    if (match) publicUrl = match[0];
    if (!output.includes('Registered tunnel connection') || !publicUrl || connected) return;
    connected = true;
    clearTimeout(connectTimeout);
    console.log(`\nСсылка для друзей: ${publicUrl}`);
    console.log(`Пользователь: takhi`);
    console.log(`Пароль: ${password}`);
    console.log('Ссылка работает, пока запущена эта команда. Остановить: Ctrl+C.\n');
  };
  tunnel.stdout.on('data', observe);
  tunnel.stderr.on('data', chunk => { observe(chunk); process.stderr.write(chunk); });
  tunnel.on('error', error => { console.error(`Не удалось запустить туннель: ${error.message}`); stop(); process.exitCode = 1; });
  tunnel.on('exit', code => {
    clearTimeout(connectTimeout);
    if (!stopping) { console.error(`Туннель остановился (${code}).`); stop(); process.exitCode = 1; }
  });
});
