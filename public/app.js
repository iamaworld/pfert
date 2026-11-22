const $ = selector => document.querySelector(selector);
const state = { horses: [], selectedFile: null, selectedFeatures: null, dataUrl: null,
  dialogHorse: null, previewUrl: null, selectionId: 0, analysisId: 0, canEditCatalog: false,
  betHorseId: null, bank: 1000, casino: localStorage.getItem('takhi-casino') !== '0' };
const maxFileSize = 7 * 1024 * 1024;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

function plural(count) {
  const mod10 = count % 10, mod100 = count % 100;
  return `${count} ${mod10 === 1 && mod100 !== 11 ? 'особь' : mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14) ? 'особи' : 'особей'} в игре`;
}

function dateSpan(horse) {
  const days = horse.photos.map(photo => photo.takenAt).filter(Boolean).sort();
  if (!days.length) return null;
  return days.length === 1 ? days[0] : `${days[0]} — ${days[days.length - 1]}`;
}

function sortedPhotos(horse) {
  return [...horse.photos].sort((a, b) => String(a.takenAt || '').localeCompare(String(b.takenAt || '')));
}

async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    const local = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
    throw new Error(local ? 'Локальный сервер «Тахи» не отвечает. Запустите его и повторите распознавание.'
      : 'Не удалось подключиться к «Тахи». Проверьте, что общая ссылка ещё работает, и повторите попытку.');
  }
  if (!response.headers.get('content-type')?.includes('application/json')) {
    throw new Error(response.status === 401 ? 'Откройте сайт заново и войдите с логином и паролем.'
      : 'Сервер вернул неожиданный ответ. Обновите страницу и повторите попытку.');
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Не удалось выполнить запрос.');
  return data;
}

function validateFile(file) {
  if (!file) throw new Error('Выберите фотографию.');
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error('Поддерживаются JPG, PNG и WebP.');
  if (file.size > maxFileSize) throw new Error('Файл слишком большой. Максимум 7 МБ.');
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Не удалось открыть изображение.')); };
    img.src = url;
  });
}

function pixels(img, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
  // Crop to the middle: the horse should occupy most of the reference frame.
  const side = Math.min(img.naturalWidth, img.naturalHeight);
  ctx.drawImage(img, (img.naturalWidth - side) / 2, (img.naturalHeight - side) / 2, side, side, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height).data;
}

function luma(data, index) { return (0.2126 * data[index] + 0.7152 * data[index + 1] + 0.0722 * data[index + 2]) / 255; }

async function extractFeatures(file) {
  validateFile(file);
  const img = await loadImage(file);
  const hashPixels = pixels(img, 9, 8);
  let bits = 0n;
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    bits = (bits << 1n) | BigInt(luma(hashPixels, (y * 9 + x) * 4) > luma(hashPixels, (y * 9 + x + 1) * 4));
  }
  const sample = pixels(img, 64, 64);
  const histogram = Array(24).fill(0);
  const grid = Array(64).fill(0);
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
    const i = (y * 64 + x) * 4;
    for (let channel = 0; channel < 3; channel++) histogram[channel * 8 + Math.min(7, Math.floor(sample[i + channel] / 32))]++;
    grid[Math.floor(y / 8) * 8 + Math.floor(x / 8)] += luma(sample, i);
  }
  return { hash: bits.toString(16).padStart(16, '0'), histogram: histogram.map(n => n / 4096), grid: grid.map(n => n / 64) };
}

function fileDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Не удалось прочитать файл.'));
    reader.readAsDataURL(file);
  });
}

const RED_NUMBERS = new Set([1, 3, 5, 7, 9, 12, 14]);
let rouletteAngle = 0;
let rouletteRaf = 0;

function numberColor(number) {
  return number === 0 ? '#0f5c43' : RED_NUMBERS.has(number) ? '#d93025' : '#0d0d0d';
}

function rouletteMarkup() {
  const sectors = state.horses.length + 1;
  const step = 360 / sectors;
  const gradient = Array.from({ length: sectors }, (_, i) => {
    const color = numberColor(i);
    return `${color} ${i * step}deg ${(i + 1) * step}deg`;
  }).join(',');
  const labels = Array.from({ length: sectors }, (_, i) => {
    const angle = i * step + step / 2;
    return `<span class="roulette-label" style="transform: rotate(${angle}deg) translateY(-82px);">${i}</span>`;
  }).join('');
  return `<div class="roulette" id="roulette"><div class="roulette-pin"></div><div class="roulette-wheel" id="roulette-wheel" style="background: conic-gradient(${gradient});">${labels}</div><div class="roulette-ball" id="roulette-ball"></div><div class="roulette-hub"></div></div>`;
}

function betLayoutHtml() {
  if (!state.horses.length) return '<span class="bet-empty">Конюшня пуста — ставки не принимаются.</span>';
  const cells = state.horses.map((horse, index) => {
    const number = index + 1;
    return `<button type="button" class="bet-cell ${RED_NUMBERS.has(number) ? 'red' : 'black'}" data-bet="${horse.id}"><b>${number}</b><span>${escapeHtml(horse.name)}</span></button>`;
  }).join('');
  const list = state.horses.map((horse, index) => {
    const number = index + 1;
    return `<div class="bet-list-row"><span class="bet-list-num ${RED_NUMBERS.has(number) ? 'red' : 'black'}">${number}</span><span>${escapeHtml(horse.name)}</span></div>`;
  }).join('');
  return `<div class="betting-layout"><div class="betting-grid"><button type="button" class="bet-cell zero" data-bet="0"><b>0</b></button>${cells}</div><div class="betting-list"><h4>Нумерация</h4>${list}</div></div>`;
}

function updateBetUi() {
  document.querySelectorAll('[data-bet]').forEach(cell => {
    const selected = cell.dataset.bet === state.betHorseId;
    cell.classList.toggle('selected', selected);
    cell.querySelector('.chip-marker')?.remove();
    if (selected) {
      const marker = document.createElement('span');
      marker.className = 'chip-marker';
      marker.textContent = '♞';
      cell.append(marker);
    }
  });
  const zero = document.querySelector('[data-bet="0"]');
  zero?.classList.toggle('selected', state.betHorseId === '0');
  const bet = state.horses.find(horse => horse.id === state.betHorseId);
  const hint = $('#bet-hint');
  if (hint) {
    hint.textContent = state.betHorseId === '0'
      ? 'Ваша ставка: 0 — не лошадь Пржевальского или неизвестная особь. Крутите барабан!'
      : bet ? `Ваша ставка: ${bet.name}. Крутите барабан!` : 'Выберите поле на столе или 0, если не знаете';
  }
  const spin = $('#spin-button');
  if (spin) spin.disabled = !state.betHorseId;
  $('#bank').textContent = String(state.bank).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

function updateCasinoUi() {
  const toggle = $('#casino-toggle');
  if (toggle) toggle.textContent = state.casino ? 'Режим: казино 🎰' : 'Режим: просто 🐴';
  const bank = document.querySelector('.bank-pill');
  if (bank) bank.hidden = !state.casino;
}

function runIdentify() {
  if (!state.selectedFile || !state.selectedFeatures || !state.dataUrl) return;
  const analysisId = ++state.analysisId;
  $('#upload-message').textContent = 'Распознаём снимок…';
  $('#result-panel').innerHTML = '<div class="result-empty"><div class="scan-mark" aria-hidden="true"><span></span><span></span><span></span><span></span></div><h3>Распознаём…</h3><p>Определяем вид и ищем особь в каталоге.</p></div>';
  request('/api/identify', { method: 'POST', body: JSON.stringify({
    features: state.selectedFeatures, dataUrl: state.dataUrl }) })
    .then(data => {
      if (analysisId !== state.analysisId) return;
      renderResults(data);
      $('#identify-button').hidden = false;
      $('#identify-button').disabled = false;
      $('#identify-button').firstChild.textContent = 'Проверить снова ';
      $('#upload-message').textContent = '';
    })
    .catch(error => {
      if (analysisId !== state.analysisId) return;
      $('#upload-message').textContent = error.message;
      $('#result-panel').innerHTML = `<div class="result-empty"><h3>Не удалось распознать снимок</h3><p>${escapeHtml(error.message)} Попробуйте ещё раз или выберите другой файл.</p></div>`;
      $('#identify-button').hidden = false;
      $('#identify-button').disabled = false;
    });
}

function showBettingTable() {
  if (!state.selectedFile || !state.selectedFeatures || !state.dataUrl) return;
  $('#result-panel').innerHTML = `<div class="result-empty bet-table">${rouletteMarkup()}<div class="jackpot-strip"><span>♠</span><span>СТОЛ ОТКРЫТ</span><span>♦</span></div><h3>Делайте ставку</h3><p>Выберите номер на столе — фишка появится на поле. 0 — если не знаете.</p>${betLayoutHtml()}<p class="bet-hint" id="bet-hint">Выберите поле на столе или 0, если не знаете</p><button id="spin-button" class="primary-button wide" type="button" disabled>КРУТИТЬ БАРАБАН <span aria-hidden="true">♞</span></button></div>`;
  updateBetUi();
}

function spinStateMarkup() {
  return `<div class="result-empty">${rouletteMarkup()}<div class="jackpot-strip"><span>♠</span><span>СТАВКА ПРИНЯТА</span><span>♦</span></div><h3>Шарик пошёл…</h3><p>Крупье-нейросеть определяет вид и ищет выигрышный номер.</p></div>`;
}

function startRoulette() {
  const wheel = $('#roulette-wheel');
  if (!wheel) return;
  cancelAnimationFrame(rouletteRaf);
  wheel.style.transition = 'none';
  const ball = $('#roulette-ball');
  ball?.classList.add('orbiting');
  ensureAudio();
  spinAudio?.pause();
  spinAudio = new Audio('/jackpot.mp3');
  spinAudio.loop = true;
  spinAudio.play().catch(() => {});
  const step = () => {
    rouletteAngle = (rouletteAngle + 7) % 1296000;
    wheel.style.transform = `rotate(${rouletteAngle}deg)`;
    rouletteRaf = requestAnimationFrame(step);
  };
  rouletteRaf = requestAnimationFrame(step);
}

function stopRoulette(sectorNumber) {
  const wheel = $('#roulette-wheel');
  if (!wheel) return Promise.resolve();
  cancelAnimationFrame(rouletteRaf);
  const ball = $('#roulette-ball');
  if (ball) {
    ball.classList.remove('orbiting');
    ball.style.transition = 'transform .6s cubic-bezier(.15,.85,.2,1)';
    ball.style.transform = 'rotate(0deg) translateX(86px)';
  }
  const sectors = state.horses.length + 1;
  const step = 360 / sectors;
  const sectorCenter = sectorNumber * step + step / 2;
  const remainder = ((-sectorCenter) % 360 + 360) % 360;
  let target = Math.floor(rouletteAngle / 360) * 360 + remainder;
  while (target <= rouletteAngle) target += 360;
  target += 3 * 360;
  rouletteAngle = target;
  wheel.style.transition = 'transform 2.2s cubic-bezier(.15,.85,.2,1)';
  wheel.style.transform = `rotate(${target}deg)`;
  return new Promise(resolve => setTimeout(() => {
    spinAudio?.pause();
    spinAudio = null;
    resolve();
  }, 2300));
}

let fxRaf = 0;
let spinAudio = null;
let audioContext = null;

function ensureAudio() {
  try {
    if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === 'suspended') audioContext.resume();
    return audioContext;
  } catch {
    return null;
  }
}

function playBombSound() {
  const ctx = ensureAudio();
  if (!ctx) return;
  const t = ctx.currentTime;
  // Transient crack.
  const crack = ctx.createBufferSource();
  const crackBuffer = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.15), ctx.sampleRate);
  const crackData = crackBuffer.getChannelData(0);
  for (let i = 0; i < crackData.length; i++) crackData[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / crackData.length, 2);
  crack.buffer = crackBuffer;
  const crackGain = ctx.createGain();
  crackGain.gain.setValueAtTime(0.9, t);
  crackGain.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
  crack.connect(crackGain).connect(ctx.destination);
  crack.start(t);
  // Deep rumble.
  const osc = ctx.createOscillator();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(95, t);
  osc.frequency.exponentialRampToValueAtTime(22, t + 3);
  const oscGain = ctx.createGain();
  oscGain.gain.setValueAtTime(0.55, t);
  oscGain.gain.exponentialRampToValueAtTime(0.001, t + 3);
  osc.connect(oscGain).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + 3);
  // Long noise wash (the blast).
  const noise = ctx.createBufferSource();
  const noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * 3, ctx.sampleRate);
  const noiseData = noiseBuffer.getChannelData(0);
  for (let i = 0; i < noiseData.length; i++) noiseData[i] = Math.random() * 2 - 1;
  noise.buffer = noiseBuffer;
  const filter = ctx.createBiquadFilter();
  filter.type = 'lowpass';
  filter.frequency.setValueAtTime(1400, t);
  filter.frequency.exponentialRampToValueAtTime(60, t + 3);
  const noiseGain = ctx.createGain();
  noiseGain.gain.setValueAtTime(0.6, t);
  noiseGain.gain.exponentialRampToValueAtTime(0.001, t + 3);
  noise.connect(filter).connect(noiseGain).connect(ctx.destination);
  noise.start(t);
  noise.stop(t + 3);
}

function playCrowdSound() {
  const ctx = ensureAudio();
  if (!ctx) return;
  const t = ctx.currentTime;
  // Disappointed "oooooh" voices.
  for (let v = 0; v < 6; v++) {
    const voice = ctx.createOscillator();
    voice.type = v % 2 ? 'sawtooth' : 'triangle';
    voice.frequency.setValueAtTime(210 + Math.random() * 320, t);
    voice.frequency.exponentialRampToValueAtTime(130 + Math.random() * 150, t + 2.1);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.11, t + 0.16);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 2.2);
    voice.connect(gain).connect(ctx.destination);
    voice.start(t);
    voice.stop(t + 2.3);
  }
  // Crowd walla.
  const crowd = ctx.createBufferSource();
  const crowdBuffer = ctx.createBuffer(1, ctx.sampleRate * 2.2, ctx.sampleRate);
  const crowdData = crowdBuffer.getChannelData(0);
  let last = 0;
  for (let i = 0; i < crowdData.length; i++) {
    const white = Math.random() * 2 - 1;
    last = (last + 0.02 * white) / 1.02;
    crowdData[i] = last * 3.2;
  }
  crowd.buffer = crowdBuffer;
  const band = ctx.createBiquadFilter();
  band.type = 'bandpass';
  band.frequency.value = 700;
  band.Q.value = 0.6;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.001, t);
  gain.gain.exponentialRampToValueAtTime(0.25, t + 0.2);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 2.2);
  crowd.connect(band).connect(gain).connect(ctx.destination);
  crowd.start(t);
  crowd.stop(t + 2.3);
}

function fxOrigin() {
  const wheel = document.querySelector('#roulette-wheel');
  if (wheel) {
    const rect = wheel.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
  }
  return { x: innerWidth / 2, y: innerHeight / 2 };
}

function startBombFx() {
  const canvas = $('#fx-canvas');
  const context = canvas.getContext('2d');
  canvas.width = innerWidth;
  canvas.height = innerHeight;
  const origin = fxOrigin();
  const flash = $('#bomb-flash');
  flash.classList.remove('flash-on');
  void flash.offsetWidth;
  flash.classList.add('flash-on');
  const colors = ['#fff1c9', '#ffd27a', '#ff9c4a', '#e05a2b', '#a33b22', '#8a8a8a', '#c9c9c9', '#5a5a5a'];
  const pieces = Array.from({ length: 180 }, () => ({
    x: origin.x + (Math.random() - 0.5) * 40,
    y: origin.y + (Math.random() - 0.5) * 20,
    vx: (Math.random() - 0.5) * 1.6,
    vy: -(1 + Math.random() * 5),
    life: 0,
    maxLife: 110 + Math.random() * 130,
    size: 2 + Math.random() * 7,
    color: colors[(Math.random() * colors.length) | 0]
  }));
  // Fireworks stay alongside the mushroom: bright sparks flying outward.
  const sparkColors = ['#f2c14e', '#d93025', '#2f9e5b', '#ffffff', '#ffd66b', '#7ec8ff'];
  const sparks = Array.from({ length: 220 }, () => ({
    x: origin.x,
    y: origin.y,
    vx: 0,
    vy: 0,
    size: 3 + Math.random() * 5,
    spin: Math.random() * Math.PI,
    color: sparkColors[(Math.random() * sparkColors.length) | 0]
  }));
  const relaunchSpark = spark => {
    const angle = Math.random() * Math.PI * 2;
    const speed = 3 + Math.random() * 9;
    spark.x = origin.x;
    spark.y = origin.y;
    spark.vx = Math.cos(angle) * speed;
    spark.vy = Math.sin(angle) * speed - 2;
  };
  for (const spark of sparks) relaunchSpark(spark);
  let ring = 0;
  const maxRing = Math.max(innerWidth, innerHeight) * 1.5;
  const tick = () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (const piece of pieces) {
      piece.life++;
      piece.vy += 0.035;
      piece.vx *= 0.996;
      piece.x += piece.vx;
      piece.y += piece.vy;
      if (piece.life > piece.maxLife) {
        piece.life = 0;
        piece.x = origin.x + (Math.random() - 0.5) * 40;
        piece.y = origin.y + (Math.random() - 0.5) * 20;
        piece.vy = -(1 + Math.random() * 5);
      }
      context.globalAlpha = Math.max(0, 1 - piece.life / piece.maxLife);
      context.fillStyle = piece.color;
      context.beginPath();
      context.arc(piece.x, piece.y, piece.size * (0.6 + Math.min(1, piece.life / 40)), 0, Math.PI * 2);
      context.fill();
    }
    for (const spark of sparks) {
      spark.x += spark.vx;
      spark.y += spark.vy;
      spark.vy += 0.18;
      spark.spin += 0.1;
      if (spark.y > canvas.height + 30 || spark.x < -20 || spark.x > canvas.width + 20) relaunchSpark(spark);
      context.save();
      context.translate(spark.x, spark.y);
      context.rotate(spark.spin);
      context.fillStyle = spark.color;
      context.fillRect(-spark.size / 2, -spark.size / 2, spark.size, spark.size * 0.6);
      context.restore();
    }
    ring += 7;
    if (ring > maxRing) ring = 0;
    context.globalAlpha = Math.max(0, 1 - ring / maxRing);
    context.strokeStyle = '#ffd9a0';
    context.lineWidth = 4;
    context.beginPath();
    context.arc(origin.x, origin.y, ring, 0, Math.PI * 2);
    context.stroke();
    context.globalAlpha = 1;
    fxRaf = requestAnimationFrame(tick);
  };
  fxRaf = requestAnimationFrame(tick);
}

function startFx(kind) {
  const canvas = $('#fx-canvas');
  const context = canvas.getContext('2d');
  canvas.width = innerWidth;
  canvas.height = innerHeight;
  const win = kind === 'win';
  const origin = fxOrigin();
  const palette = win
    ? ['#f2c14e', '#d93025', '#2f9e5b', '#ffffff', '#ffd66b', '#7ec8ff']
    : ['#6b7f8f', '#8fa7b8', '#4a5a66', '#aebfca', '#2f3b45', '#d93025'];
  const pieces = Array.from({ length: win ? 220 : 90 }, () => {
    const angle = Math.random() * Math.PI * 2;
    const speed = win ? 3 + Math.random() * 9 : 0.5 + Math.random() * 1.5;
    return {
      x: win ? origin.x : Math.random() * canvas.width,
      y: win ? origin.y : -20 - Math.random() * canvas.height * 0.5,
      vx: win ? Math.cos(angle) * speed : (Math.random() - 0.5) * 0.6,
      vy: win ? Math.sin(angle) * speed - 2 : 1 + Math.random() * 2.5,
      gravity: win ? 0.18 : 0.012,
      size: 3 + Math.random() * 5,
      spin: Math.random() * Math.PI,
      color: palette[(Math.random() * palette.length) | 0]
    };
  });
  const tick = () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (const piece of pieces) {
      piece.x += piece.vx;
      piece.y += piece.vy;
      piece.vy += piece.gravity;
      piece.spin += 0.1;
      if (win) {
        if (piece.y > canvas.height + 30 || piece.x < -20 || piece.x > canvas.width + 20) {
          piece.x = origin.x;
          piece.y = origin.y;
          const angle = Math.random() * Math.PI * 2;
          const speed = 3 + Math.random() * 9;
          piece.vx = Math.cos(angle) * speed;
          piece.vy = Math.sin(angle) * speed - 2;
        }
      } else {
        if (piece.y > canvas.height + 30) {
          piece.y = -20;
          piece.x = Math.random() * canvas.width;
          piece.vy = 1 + Math.random() * 2.5;
        }
        if (piece.x < -20) piece.x = canvas.width + 20;
        if (piece.x > canvas.width + 20) piece.x = -20;
      }
      context.save();
      context.translate(piece.x, piece.y);
      context.rotate(piece.spin);
      context.fillStyle = piece.color;
      if (win) {
        context.fillRect(-piece.size / 2, -piece.size / 2, piece.size, piece.size * 0.6);
      } else {
        context.fillRect(-piece.size / 4, -piece.size, piece.size / 2, piece.size * 2);
      }
      context.restore();
    }
    fxRaf = requestAnimationFrame(tick);
  };
  fxRaf = requestAnimationFrame(tick);
}

function closeFx() {
  cancelAnimationFrame(fxRaf);
  const overlay = $('#fx-overlay');
  overlay.hidden = true;
  $('#bomb-flash')?.classList.remove('flash-on');
  const canvas = $('#fx-canvas');
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function showFx(kind, title, sub) {
  const overlay = $('#fx-overlay');
  overlay.className = `fx-overlay ${kind}`;
  overlay.hidden = false;
  $('#fx-title').textContent = title;
  $('#fx-sub').textContent = sub;
  cancelAnimationFrame(fxRaf);
  if (kind === 'win') {
    startBombFx();
    playBombSound();
  } else {
    startFx(kind);
    playCrowdSound();
  }
}

function bankDelta(delta) {
  state.bank = Math.max(0, state.bank + delta);
  updateBetUi();
}

function renderCatalog() {
  $('#horse-count').textContent = plural(state.horses.length);
  $('#catalog-number').textContent = String(state.horses.length).padStart(2, '0');
  const grid = $('#catalog-grid');
  $('#add-horse').hidden = !state.canEditCatalog;
  if (!state.horses.length) {
    grid.innerHTML = state.canEditCatalog
      ? '<div class="catalog-empty"><strong>Конюшня пока пуста</strong>Добавьте первую особь с опорной фотографией, чтобы открыть приём ставок.</div>'
      : '<div class="catalog-empty"><strong>Конюшня пока пуста</strong>Владелец ещё не добавил опорные снимки. Определение вида доступно.</div>';
    return;
  }
  grid.innerHTML = state.horses.map(horse => {
    const span = dateSpan(horse);
    return `<article class="horse-card"><img src="${escapeHtml(horse.photos[0].url)}" alt="${escapeHtml(horse.name)}"><div class="horse-card-body"><div class="horse-card-top"><h3>${escapeHtml(horse.name)}</h3><small>${horse.photos.length} фото${span ? `<br>${escapeHtml(span)}` : ''}</small></div><p>${escapeHtml(horse.notes || 'Приметы пока не указаны')}</p><div class="horse-card-actions"><button class="subtle-button" type="button" data-timeline="${horse.id}">Хронология забегов</button><button class="subtle-button" type="button" data-pedigree="${horse.id}">Родословная</button>${state.canEditCatalog ? `<button class="subtle-button" type="button" data-add-photo="${horse.id}">Добавить ракурс</button><button class="subtle-button danger" type="button" data-delete="${horse.id}">Удалить</button>` : ''}</div></div></article>`;
  }).join('');
}

function reviewControls(winner = null) {
  if (!state.canEditCatalog || !state.horses.length) return '';
  const options = state.horses.map(horse =>
    `<option value="${escapeHtml(horse.id)}">${escapeHtml(horse.name)}</option>`).join('');
  return `<section class="identity-review" aria-label="Разметка снимка" data-suggested="${escapeHtml(winner?.horse.id ?? '')}">
    ${winner ? `<button type="button" class="primary-button" data-confirm-horse="${escapeHtml(winner.horse.id)}">Подтвердить особь</button>` : ''}
    <details ${winner ? '' : 'open'}>
      <summary>${winner ? 'Выбрать другую особь' : 'Указать особь из каталога'}</summary>
      <form class="identity-correction">
        <label>Особь на снимке <select name="horseId" required><option value="">Выберите особь</option>${options}</select></label>
        <button class="outline-button" type="submit">Сохранить выбор</button>
      </form>
    </details>
    <p class="review-status" role="status"></p>
  </section>`;
}

function renderResults({ candidates, method, species, classification = [] }) {
  const panel = $('#result-panel');
  if (species?.decision === 'no_horse_detected') {
    panel.innerHTML = '<div class="result-content verdict-block"><span class="verdict-label">Барабан пуст</span><h3>Лошадь не обнаружена</h3><p>Если лошадь на фото есть, попробуйте более крупный или чёткий снимок.</p></div>';
    return;
  }
  const qualityNote = species?.lowQuality
    ? '<p class="quality-note">Снимок тёмный или размытый. Крупье мог ошибиться.</p>' : '';
  if (species?.decision === 'not_przewalski') {
    panel.innerHTML = `<div class="result-content verdict-block"><span class="verdict-label">Результат по виду</span><h3>Не лошадь Пржевальского</h3><p>Крупье-нейросеть отнесла снимок к другим животным. На сложных кадрах возможна ошибка.</p>${qualityNote}${reviewControls()}</div>`;
    return;
  }
  const speciesNotice = species?.decision === 'przewalski'
    ? '<div class="species-notice"><strong>♞ Лошадь Пржевальского</strong><span>Вид определён автоматически по фотографии.</span></div>'
    : '<div class="species-notice review"><strong>Вид не определён</strong><span>Доступно только сравнение с фотографиями конюшни.</span></div>';
  if (!candidates.length) {
    const description = state.horses.length
      ? 'Колесо не нашло кандидатов. Если вы узнали лошадь, укажите её вручную.'
      : species?.decision === 'przewalski'
      ? 'В конюшне нет эталонов. Вид определён, но назвать конкретную лошадь без эталонов нельзя.'
      : 'В конюшне нет эталонов. Добавьте их для сравнения; определение вида сейчас недоступно.';
    panel.innerHTML = `<div class="result-content">${speciesNotice}${qualityNote}<div class="result-empty no-identity"><h3>Ставка без выигрыша</h3><p>${description}</p>${state.canEditCatalog && !state.horses.length ? '<button class="outline-button" id="add-known-from-result" type="button">Добавить известную особь</button>' : ''}</div>${reviewControls()}</div>`;
    return;
  }
  const learned = method === 'trained' || method === 'megadescriptor';
  const isVision = method === 'vision';
  const label = method === 'trained' ? 'Дообученный крупье' : learned ? 'Крупье-нейросеть' : isVision ? 'Apple Vision' : 'Базовое сравнение';
  const winner = candidates[0];
  const others = candidates.slice(1);
  const classBars = classification.length ? `<div class="class-bars"><h4>Коэффициенты конюшни</h4>${classification.slice(0, 6).map((item, index) => {
    const pct = index === 0 ? 100 : Math.min(99.9, Math.round(item.similarity * 1000) / 10);
    return `<div class="class-bar"><span>${escapeHtml(item.name)}</span><div class="class-track"><i style="width:${Math.max(2, pct)}%"></i></div><em>${pct.toFixed(1)}%</em></div>`;
  }).join('')}<p class="class-note">100% — ближайшее совпадение, остальные — относительно него. Это сходство, а не вероятность.</p></div>` : '';
  panel.innerHTML = `<div class="result-content">${speciesNotice}${qualityNote}<div class="identity-heading"><span class="verdict-label">Выигрышная особь</span><span class="method-label">${label}</span></div><div class="identity-match"><img src="${escapeHtml(winner.photoUrl)}" alt="Опорный снимок ${escapeHtml(winner.horse.name)}"><div><span>Барабан показал</span><h3>${escapeHtml(winner.horse.name)}</h3><p>${winner.horse.photos.length} фото в конюшне</p><button class="subtle-button" type="button" data-open-timeline="${winner.horse.id}">Хронология забегов</button></div></div>${others.length ? `<div class="alternatives"><h4>Другие фавориты</h4>${others.map(item => `<div class="candidate"><img src="${escapeHtml(item.photoUrl)}" alt="Опорный снимок ${escapeHtml(item.horse.name)}"><div><strong>${escapeHtml(item.horse.name)}</strong><small>${item.horse.photos.length} фото в конюшне</small></div></div>`).join('')}</div>` : ''}${classBars}<p class="result-advice">Крупье выбрал ближайшую запись автоматически. Точность определения конкретной особи ещё не измерена; если нужной особи нет в конюшне, барабан всё равно покажет ближайшую.</p></div>`;
  panel.querySelector('.result-content').insertAdjacentHTML('beforeend', reviewControls(winner));
}

async function runSpin() {
  if (!state.selectedFile || !state.selectedFeatures || !state.dataUrl || !state.betHorseId) return;
  const analysisId = ++state.analysisId;
  $('#upload-message').textContent = 'Крупье принял ставку…';
  $('#result-panel').innerHTML = spinStateMarkup();
  startRoulette();
  try {
    const data = await request('/api/identify', { method: 'POST', body: JSON.stringify({
      features: state.selectedFeatures, dataUrl: state.dataUrl }) });
    if (analysisId !== state.analysisId) return;
    const target = $('#horse-targets button[aria-pressed="true"]');
    if (target) {
      const name = data.species?.decision !== 'not_przewalski' && data.candidates[0]?.horse.name;
      target.textContent = name ? `${target.dataset.label}: ${name}?` : target.dataset.label;
    }
    const winner = data.candidates[0]?.horse;
    const winnerIndex = winner ? state.horses.findIndex(horse => horse.id === winner.id) : -1;
    const sector = winnerIndex >= 0 ? winnerIndex + 1 : 0;
    await stopRoulette(sector);
    if (analysisId !== state.analysisId) return;
    if (state.betHorseId === '0') {
      const notHorse = data.species?.decision === 'not_przewalski';
      const unknown = !winner && data.species?.decision !== 'przewalski';
      if (notHorse || unknown) {
        bankDelta(500);
        showFx('win', 'ДЖЕКПОТ!', 'Вы поставили на 0 — и это не лошадь Пржевальского (или особь неизвестна). Нажмите, чтобы закрыть.');
      } else {
        bankDelta(-150);
        showFx('lose', 'СТАВКА СГОРЕЛА', `Вы поставили на 0, а крупье узнал ${winner.name}. Нажмите, чтобы закрыть.`);
      }
    } else if (state.betHorseId && winner) {
      if (winner.id === state.betHorseId) {
        bankDelta(500);
        showFx('win', 'ДЖЕКПОТ!', `Вы угадали: ${winner.name}. Банк пополнен на 500. Нажмите, чтобы закрыть.`);
      } else {
        const bet = state.horses.find(horse => horse.id === state.betHorseId);
        bankDelta(-150);
        showFx('lose', 'СТАВКА СГОРЕЛА', `Вы ставили на ${bet?.name || '—'}, а шарик показал ${winner.name}. Нажмите, чтобы закрыть.`);
      }
    }
    renderResults(data);
    $('#identify-button').hidden = false;
    $('#identify-button').disabled = false;
    $('#identify-button').firstChild.textContent = 'Сыграть ещё раз ';
    $('#upload-message').textContent = '';
  } catch (error) {
    if (analysisId !== state.analysisId) return;
    $('#upload-message').textContent = error.message;
    $('#result-panel').innerHTML = `<div class="result-empty"><h3>Ставка не сыграла</h3><p>${escapeHtml(error.message)} Попробуйте ещё раз или выберите другой файл.</p></div>`;
    $('#identify-button').hidden = false;
    $('#identify-button').disabled = false;
  }
}

function cropHorse(image, box) {
  const paddingX = box.width * 0.08;
  const paddingY = box.height * 0.08;
  const left = Math.max(0, box.x - paddingX) * image.naturalWidth;
  const top = Math.max(0, box.y - paddingY) * image.naturalHeight;
  const width = Math.min(image.naturalWidth, (box.x + box.width + paddingX) * image.naturalWidth) - left;
  const height = Math.min(image.naturalHeight, (box.y + box.height + paddingY) * image.naturalHeight) - top;
  const scale = Math.min(1, 1600 / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext('2d');
  context.fillStyle = '#fff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, left, top, width, height, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob(blob => {
    if (blob) resolve(blob);
    else reject(new Error('Не удалось вырезать лошадь из снимка.'));
  }, 'image/jpeg', 0.92));
}

async function analyzeTarget(file, box, index = -1) {
  const id = ++state.analysisId;
  state.dataUrl = null;
  state.selectedFeatures = null;
  $('#identify-button').disabled = true;
  $('#identify-button').hidden = true;
  $('#result-panel').textContent = index < 0 ? 'Крупье готовит стол…' : `Крупье готовит фрагмент «Лошадь ${index + 1}»…`;
  for (const button of $('#photo-stage').querySelectorAll('[data-target]')) {
    button.setAttribute('aria-pressed', String(Number(button.dataset.target) === index));
  }
  try {
    const fragment = box ? await cropHorse($('#preview'), box) : file;
    const [features, dataUrl] = await Promise.all([extractFeatures(fragment), fileDataUrl(fragment)]);
    if (id !== state.analysisId) return;
    state.selectedFile = file;
    state.selectedFeatures = features;
    state.dataUrl = dataUrl;
    if (state.casino) showBettingTable();
    else runIdentify();
  } catch (error) {
    if (id !== state.analysisId) return;
    $('#result-panel').textContent = error.message;
  }
}

function showHorseTargets(file, boxes) {
  const overlay = $('#horse-boxes');
  const targets = $('#horse-targets');
  targets.hidden = !boxes.length;
  boxes.forEach((box, index) => {
    const label = `Лошадь ${index + 1}`;
    const frame = document.createElement('button');
    frame.type = 'button';
    frame.className = 'horse-box';
    frame.setAttribute('aria-label', `Рамка: ${label}`);
    frame.dataset.target = index;
    Object.assign(frame.style, {left: `${box.x * 100}%`, top: `${box.y * 100}%`,
      width: `${box.width * 100}%`, height: `${box.height * 100}%`});
    const number = document.createElement('span');
    number.textContent = String(index + 1);
    frame.append(number);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'outline-button';
    button.textContent = label;
    button.dataset.label = label;
    button.dataset.target = index;
    for (const control of [frame, button]) {
      control.setAttribute('aria-pressed', 'false');
      control.addEventListener('click', () => analyzeTarget(file, box, index));
    }
    overlay.append(frame);
    targets.append(button);
  });
}

async function selectPhoto(file) {
  if (!file) return;
  try {
    validateFile(file);
  } catch (error) {
    $('#upload-message').textContent = error.message;
    return;
  }
  const selectionId = ++state.selectionId;
  try {
    state.analysisId++;
    state.selectedFile = null;
    state.dataUrl = null;
    state.betHorseId = null;
    $('#horse-boxes').replaceChildren();
    $('#horse-targets').replaceChildren();
    $('#horse-targets').hidden = true;
    $('#photo-hint').textContent = '';
    $('#result-panel').textContent = 'Крупье принимает ставку…';
    $('#identify-button').disabled = true;
    $('#identify-button').hidden = true;
    $('#upload-message').textContent = 'Крупье принимает ставку…';
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);
    const preview = $('#preview');
    preview.src = state.previewUrl;
    const partsOverlay = $('#parts-overlay');
    partsOverlay.hidden = true;
    partsOverlay.removeAttribute('src');
    const partsButton = $('#parts-button');
    partsButton.hidden = false;
    partsButton.textContent = 'Части тела';
    $('#photo-stage').hidden = false;
    $('#dropzone').hidden = true;
    const [dataUrl] = await Promise.all([fileDataUrl(file), preview.decode()]);
    if (selectionId !== state.selectionId) return;
    $('#upload-message').textContent = 'Крупье ищет лошадей на снимке…';
    const detection = await request('/api/detect-horses', { method: 'POST', body: JSON.stringify({dataUrl}) })
      .catch(() => ({available: false, boxes: []}));
    if (selectionId !== state.selectionId) return;
    showHorseTargets(file, detection.boxes);
    const hint = $('#photo-hint');
    hint.textContent = detection.boxes.length ? 'Поставьте на лошадь: нажмите на рамку или номер.'
      : detection.available ? 'На исходном снимке нет найденных лошадей.'
        : 'Не удалось получить рамки. Крупье повторит проверку.';
    await analyzeTarget(file, detection.boxes[0], detection.boxes.length ? 0 : -1);
  } catch (error) {
    if (selectionId !== state.selectionId) return;
    $('#upload-message').textContent = error.message;
    $('#result-panel').innerHTML = `<div class="result-empty"><h3>Не удалось открыть фото</h3><p>${escapeHtml(error.message)}</p></div>`;
  }
}

const dropzone = $('#dropzone');
dropzone.addEventListener('click', () => $('#photo-input').click());
dropzone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('#photo-input').click(); } });
$('#change-photo').addEventListener('click', event => { event.stopPropagation(); $('#photo-input').click(); });
$('#photo-input').addEventListener('change', event => {
  const file = event.target.files[0];
  event.target.value = '';
  selectPhoto(file);
});
const uploadPanel = dropzone.closest('.upload-panel');
for (const eventName of ['dragenter', 'dragover']) uploadPanel.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.add('dragging'); });
for (const eventName of ['dragleave', 'drop']) uploadPanel.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.remove('dragging'); });
uploadPanel.addEventListener('drop', event => selectPhoto(event.dataTransfer.files[0]));

$('#identify-button').addEventListener('click', () => {
  if (state.casino) showBettingTable();
  else runIdentify();
});
$('#casino-toggle').addEventListener('click', () => {
  state.casino = !state.casino;
  localStorage.setItem('takhi-casino', state.casino ? '1' : '0');
  updateCasinoUi();
});
$('#result-panel').addEventListener('click', event => {
  const betCell = event.target.closest('[data-bet]');
  if (betCell) {
    if (betCell.dataset.bet === '0') state.betHorseId = state.betHorseId === '0' ? null : '0';
    else state.betHorseId = state.betHorseId === betCell.dataset.bet ? null : betCell.dataset.bet;
    updateBetUi();
    return;
  }
  if (event.target.closest('#spin-button')) {
    runSpin();
    return;
  }
});
$('#fx-overlay').addEventListener('click', closeFx);
async function saveReview(section, horseId) {
  if (!state.dataUrl || section.dataset.saving) return;
  section.dataset.saving = 'true';
  const controls = [...section.querySelectorAll('button, select')];
  const status = section.querySelector('.review-status');
  controls.forEach(control => { control.disabled = true; });
  status.textContent = 'Сохраняем разметку…';
  try {
    const result = await request('/api/identity-reviews', { method: 'POST', body: JSON.stringify({
      horseId, suggestedHorseId: section.dataset.suggested || null, dataUrl: state.dataUrl
    }) });
    status.textContent = `Разметка сохранена: ${result.horseName}.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    delete section.dataset.saving;
    controls.forEach(control => { control.disabled = false; });
  }
}

$('#result-panel').addEventListener('click', event => {
  if (event.target.closest('#add-known-from-result')) openDialog();
  const timelineButton = event.target.closest('[data-open-timeline]');
  if (timelineButton) {
    const horse = state.horses.find(item => item.id === timelineButton.dataset.openTimeline);
    if (horse) openTimeline(horse);
    return;
  }
  const button = event.target.closest('[data-confirm-horse]');
  if (button) saveReview(button.closest('.identity-review'), button.dataset.confirmHorse);
});
$('#result-panel').addEventListener('submit', event => {
  if (!event.target.matches('.identity-correction')) return;
  event.preventDefault();
  saveReview(event.target.closest('.identity-review'), new FormData(event.target).get('horseId'));
});

const dialog = $('#horse-dialog');
function openDialog(horse = null) {
  if (!state.canEditCatalog) return;
  state.dialogHorse = horse;
  $('#horse-form').reset();
  $('#dialog-message').textContent = '';
  $('#horse-file-name').textContent = 'Выбрать снимок';
  $('#dialog-title').textContent = horse ? `Новый ракурс: ${horse.name}` : 'Добавить особь';
  $('#dialog-description').textContent = horse ? 'Дополнительный снимок поможет сравнивать новые ракурсы.' : 'Добавьте имя или полевой номер и опорный снимок. Позже можно добавить другие ракурсы.';
  $('#horse-name').closest('label').hidden = !!horse;
  $('#horse-notes').closest('label').hidden = !!horse;
  $('#horse-name').required = !horse;
  $('#save-horse').textContent = horse ? 'Сохранить снимок' : 'Сохранить запись';
  dialog.showModal();
}
$('#add-horse').addEventListener('click', () => openDialog());
$('#close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
$('#horse-photo').addEventListener('change', event => { $('#horse-file-name').textContent = event.target.files[0]?.name || 'Выбрать снимок'; });
$('#catalog-grid').addEventListener('click', async event => {
  const timelineButton = event.target.closest('[data-timeline]');
  if (timelineButton) {
    const horse = state.horses.find(item => item.id === timelineButton.dataset.timeline);
    if (horse) openTimeline(horse);
    return;
  }
  const pedigreeButton = event.target.closest('[data-pedigree]');
  if (pedigreeButton) {
    window.openPedigree?.(pedigreeButton.dataset.pedigree);
    return;
  }
  const addButton = event.target.closest('[data-add-photo]');
  if (addButton) return openDialog(state.horses.find(horse => horse.id === addButton.dataset.addPhoto));
  const deleteButton = event.target.closest('[data-delete]');
  if (!deleteButton) return;
  const horse = state.horses.find(item => item.id === deleteButton.dataset.delete);
  if (!horse || !confirm(`Удалить запись «${horse.name}» и все её фотографии?`)) return;
  try {
    await request(`/api/horses/${horse.id}`, { method: 'DELETE' });
    state.horses = state.horses.filter(item => item.id !== horse.id);
    if (state.betHorseId === horse.id) state.betHorseId = null;
    renderCatalog();
    document.dispatchEvent(new CustomEvent('catalog-updated', { detail: state.horses }));
    if (state.selectedFile) showBettingTable();
  } catch (error) { alert(error.message); }
});

const timelineDialog = $('#timeline-dialog');
function openTimeline(horse) {
  $('#timeline-title').textContent = horse.name;
  $('#timeline-notes').textContent = horse.notes || 'Приметы пока не указаны';
  const photos = sortedPhotos(horse);
  $('#timeline-body').innerHTML = photos.length
    ? photos.map(photo => `<figure class="timeline-figure"><img src="${escapeHtml(photo.url)}" alt="${escapeHtml(horse.name)}"><figcaption>${escapeHtml(photo.takenAt || 'дата неизвестна')}</figcaption></figure>`).join('')
    : '<p class="timeline-empty">Фотографий пока нет.</p>';
  timelineDialog.showModal();
}
$('#close-timeline').addEventListener('click', () => timelineDialog.close());
timelineDialog.addEventListener('click', event => { if (event.target === timelineDialog) timelineDialog.close(); });

async function showParts() {
  if (!state.dataUrl) return;
  const button = $('#parts-button');
  const overlay = $('#parts-overlay');
  if (!overlay.hidden) { overlay.hidden = true; return; }
  button.disabled = true;
  button.textContent = 'Сегментируем…';
  try {
    const result = await request('/api/parts', { method: 'POST', body: JSON.stringify({ dataUrl: state.dataUrl }) });
    if (result.available) {
      overlay.src = result.overlayDataUrl;
      overlay.hidden = false;
      const confidence = result.confidence || {};
      $('#photo-hint').textContent = `Части тела: уши ${Math.round((confidence.ears || 0) * 100)}% · ноги ${Math.round((confidence.legs || 0) * 100)}% · тело ${Math.round((confidence.body || 0) * 100)}%`;
    } else {
      $('#photo-hint').textContent = result.reason || 'Сегментация частей тела недоступна.';
    }
  } catch (error) {
    $('#photo-hint').textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = overlay.hidden ? 'Части тела' : 'Скрыть сегментацию';
  }
}
$('#parts-button').addEventListener('click', showParts);

$('#horse-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = $('#save-horse');
  try {
    const file = $('#horse-photo').files[0];
    validateFile(file);
    button.disabled = true;
    $('#dialog-message').textContent = 'Обрабатываем…';
    const features = await extractFeatures(file);
    const photo = { dataUrl: await fileDataUrl(file), features };
    const horse = state.dialogHorse;
    const url = horse ? `/api/horses/${horse.id}/photos` : '/api/horses';
    const body = horse ? { photo } : { name: $('#horse-name').value, notes: $('#horse-notes').value, photo };
    const data = await request(url, { method: 'POST', body: JSON.stringify(body) });
    if (horse) state.horses = state.horses.map(item => item.id === horse.id ? data.horse : item);
    else state.horses.unshift(data.horse);
    renderCatalog(); dialog.close();
    document.dispatchEvent(new CustomEvent('catalog-updated', { detail: state.horses }));
    if (state.selectedFile) showBettingTable();
  } catch (error) { $('#dialog-message').textContent = error.message; }
  finally { button.disabled = false; }
});

document.addEventListener('catalog-updated', event => {
  if (Array.isArray(event.detail)) {
    state.horses = event.detail;
    renderCatalog();
  }
});

try {
  const [config, catalog] = await Promise.all([request('/api/config'), request('/api/horses')]);
  state.canEditCatalog = config.canEditCatalog;
  state.horses = catalog.horses;
  renderCatalog();
  updateBetUi();
  updateCasinoUi();
}
catch (error) { $('#catalog-grid').innerHTML = `<div class="catalog-empty">${escapeHtml(error.message)}</div>`; }
