#!/usr/bin/env node
// Автоматический запуск того же стенда в отдельном Chrome через DevTools.
// Chrome запускается с remote-debugging-port, этот файл только нажимает кнопку
// на dev-странице и забирает итоговый JSON — формулы остаются в самой странице.

const endpoint = process.argv[2] || 'http://127.0.0.1:9223';
const action = process.argv[3] || '15';
const seconds = action.startsWith('sync') ? null : Number(action);
const maxDpr = process.argv[4] ? Number(process.argv[4]) : null;
const variant = process.argv[5] || 'full';
if (!['sync', 'sync-result'].includes(action) && ![15, 180].includes(seconds)) {
  throw new Error('action must be sync, sync-result, 15 or 180 seconds');
}

const targets = await (await fetch(`${endpoint}/json/list`)).json();
const target = targets.find((item) => item.type === 'page' && item.url.includes('/avatar/dev/'));
if (!target) throw new Error('avatar dev page is not open in the debug Chrome');

const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

let id = 0;
const pending = new Map();
socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  if (!message.id || !pending.has(message.id)) return;
  const { resolve, reject } = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(message.error.message)); else resolve(message.result);
});

function command(method, params = {}) {
  return new Promise((resolve, reject) => {
    const callId = ++id;
    pending.set(callId, { resolve, reject });
    socket.send(JSON.stringify({ id: callId, method, params }));
  });
}

async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {
    expression, awaitPromise: true, returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.result?.description || 'evaluation failed');
  return result.result.value;
}

await command('Page.bringToFront');
if (action === 'sync-result') {
  const result = await evaluate(`({
    text: document.querySelector('#syncOut')?.innerText || '',
    error: document.querySelector('#err')?.innerText || '',
    audioState: window.__dev?.clock?.ctx?.state || null
  })`);
  socket.close();
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}
await command('Page.reload', { ignoreCache: true });
await new Promise((resolve) => setTimeout(resolve, 4000));
const selector = action === 'sync' ? '#measure' : `#perf${seconds}`;
const ready = await evaluate(`({
  button: !!document.querySelector('${selector}'),
  title: document.title,
  url: location.href,
  error: document.querySelector('#err')?.textContent || ''
})`);
if (!ready.button) throw new Error(`dev page did not boot: ${JSON.stringify(ready)}`);
if (action === 'sync') {
  // Реальный pointer event, а не element.click(): AudioContext по правилам
  // браузера должен стартовать только из пользовательской активации.
  const rect = await evaluate(`(() => {
    const r = document.querySelector('#measure').getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  })()`);
  await command('Input.dispatchMouseEvent', { type: 'mousePressed', x: rect.x, y: rect.y,
    button: 'left', clickCount: 1 });
  await command('Input.dispatchMouseEvent', { type: 'mouseReleased', x: rect.x, y: rect.y,
    button: 'left', clickCount: 1 });
  await new Promise((resolve) => setTimeout(resolve, 26000));
  const result = await evaluate(`({
    text: document.querySelector('#syncOut').innerText,
    error: document.querySelector('#err').innerText,
    audioState: window.__dev.clock?.ctx?.state || null
  })`);
  socket.close();
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}
if (maxDpr || variant === 'dof-quarter') await evaluate(`(() => {
  ${maxDpr ? `window.__dev.avatar.look.cfg.renderer.maxPixelRatio = ${maxDpr};` : ''}
  ${variant === 'dof-quarter' ? 'window.__dev.avatar.look.dofScale = 0.25;' : ''}
  window.__dev.avatar.setSize(innerWidth, innerHeight);
})()`);
await evaluate(`document.querySelector('#perf${seconds}').click()`);
if (variant === 'no-dof') {
  await evaluate(`window.__dev.avatar.look.bokeh.enabled = false`);
}
await new Promise((resolve) => setTimeout(resolve, (seconds + 2) * 1000));
const result = await evaluate('window.__dev.lastPerf');
socket.close();
console.log(JSON.stringify(result, null, 2));
