import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { growthRoutes } from './growth.js';
import { couldBeAlive, editPedigree, optionalDate } from './pedigree.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(root, 'public');
const dataDir = process.env.TAKHI_DATA_DIR ? path.resolve(process.env.TAKHI_DATA_DIR) : path.join(root, 'data');
const imageDir = path.join(dataDir, 'images');
const tempDir = path.join(dataDir, 'tmp');
const referenceCacheDir = path.join(dataDir, 'reference-cache');
const visionSource = path.join(root, 'vision-match.swift');
const visionBinary = path.join(dataDir, 'vision-match');
const localPython = path.join(root, '.venv-takhi', 'bin', 'python');
const mlPython = process.env.TAKHI_PYTHON || (await fs.stat(localPython).catch(() => null)
  ? localPython : path.join(root, '.venv', 'bin', 'python'));
const mlScript = path.join(root, 'ml', 'search.py');
const exifScript = path.join(root, 'ml', 'exif_date.py');
const partsScript = path.join(root, 'ml', 'parts_segment.py');
const partsWeights = path.join(dataDir, 'ml', 'horse_parts.pt');
const dinoPartsScript = path.join(root, 'ml', 'parts.py');
const dinoPartsBackbone = path.join(dataDir, 'ml', 'dinov3-backbone.pth');
const dinoPartsCheckpoint = path.join(dataDir, 'ml', 'horse_parts.pt');
const detectorScript = path.join(root, 'ml', 'detect_horses.py');
const detectorWeights = path.join(dataDir, 'ml', 'yolo26x.pt');
const mlHead = path.join(dataDir, 'ml', 'horse_head.pt');
const dbFile = path.join(dataDir, 'horses.json');
const reviewDir = path.join(dataDir, 'identity-reviews');
const port = Number(process.env.PORT || 3000);
const host = process.env.TAKHI_HOST || '127.0.0.1';
const sharePassword = process.env.TAKHI_SHARE_PASSWORD || '';
const shareMode = process.env.TAKHI_SHARE_MODE === '1';
if (host !== '127.0.0.1' && host !== '::1' && !sharePassword) {
  throw new Error('Сетевой доступ разрешён только с TAKHI_SHARE_PASSWORD.');
}
const maxBody = 10 * 1024 * 1024;
const maxActiveIdentifications = 2;
const detectorCacheLimit = 64;
const detectorCache = new Map();
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.mp3': 'audio/mpeg' };

await fs.mkdir(imageDir, { recursive: true });
await fs.mkdir(tempDir, { recursive: true });
const execFileAsync = promisify(execFile);
let visionBuild;
let horses;
try { horses = JSON.parse(await fs.readFile(dbFile, 'utf8')); }
catch (error) { if (error.code !== 'ENOENT') throw error; horses = []; }
if (!Array.isArray(horses)) throw new Error('Invalid catalog');
let writeQueue = Promise.resolve();
let activeIdentifications = 0;
let segmenting = false;

function hasShareAccess(header) {
  if (!sharePassword) return true;
  if (typeof header !== 'string' || !header.startsWith('Basic ') || header.length > 4096) return false;
  let credentials;
  try { credentials = Buffer.from(header.slice(6), 'base64').toString('utf8'); }
  catch { return false; }
  const separator = credentials.indexOf(':');
  if (separator < 0 || credentials.slice(0, separator) !== 'takhi') return false;
  const provided = crypto.createHash('sha256').update(credentials.slice(separator + 1)).digest();
  const expected = crypto.createHash('sha256').update(sharePassword).digest();
  return crypto.timingSafeEqual(provided, expected);
}

function send(res, status, body, type = 'application/json; charset=utf-8') {
  const output = type.startsWith('application/json') ? JSON.stringify(body) : body;
  res.writeHead(status, { 'Content-Type': type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
  res.end(output);
}

async function readJson(req) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBody) { const error = new Error('Файл слишком большой. Максимум 7 МБ.'); error.status = 413; throw error; }
    chunks.push(chunk);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch { const error = new Error('Некорректные данные запроса.'); error.status = 400; throw error; }
}

function validFeatures(value) {
  return value && typeof value.hash === 'string' && /^[0-9a-f]{16}$/.test(value.hash) &&
    Array.isArray(value.histogram) && value.histogram.length === 24 &&
    Array.isArray(value.grid) && value.grid.length === 64 &&
    [...value.histogram, ...value.grid].every(n => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1);
}

function decodeImage(dataUrl) {
  if (typeof dataUrl !== 'string') throw Object.assign(new Error('Добавьте фотографию.'), { status: 400 });
  const match = /^data:image\/(jpeg|png|webp);base64,([a-zA-Z0-9+/=]+)$/.exec(dataUrl);
  if (!match) throw Object.assign(new Error('Поддерживаются JPG, PNG и WebP.'), { status: 400 });
  const bytes = Buffer.from(match[2], 'base64');
  if (bytes.length < 100 || bytes.length > 7 * 1024 * 1024) throw Object.assign(new Error('Размер фотографии должен быть до 7 МБ.'), { status: 400 });
  const [kind] = match.slice(1);
  const isJpeg = bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  const isPng = bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]));
  const isWebp = bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP';
  if (!({ jpeg: isJpeg, png: isPng, webp: isWebp })[kind]) throw Object.assign(new Error('Формат файла не совпадает с изображением.'), { status: 400 });
  return { bytes, extension: kind === 'jpeg' ? 'jpg' : kind };
}

function publicHorse(horse) {
  return { id: horse.id, name: horse.name, notes: horse.notes, createdAt: horse.createdAt,
    sex: horse.sex ?? null, birthOn: horse.birthOn ?? null, deceased: horse.deceased === true,
    deathOn: horse.deathOn ?? null, motherId: horse.motherId ?? null, fatherId: horse.fatherId ?? null,
    partnerIds: horse.partnerIds ?? [],
    photos: horse.photos.map(({ id, url, createdAt, takenAt, capturedOn, foal }) =>
      ({ id, url, createdAt, takenAt: takenAt ?? capturedOn ?? null, capturedOn: capturedOn ?? null, foal: foal === true })) };
}

function persist() {
  const snapshot = JSON.stringify(horses, null, 2);
  writeQueue = writeQueue.catch(() => {}).then(async () => {
    const temp = `${dbFile}.${crypto.randomUUID()}.tmp`;
    await fs.writeFile(temp, snapshot);
    await fs.rename(temp, dbFile);
  });
  return writeQueue;
}

async function readExifDate(file) {
  if (!(await fs.stat(mlPython).catch(() => null)) || !(await fs.stat(exifScript).catch(() => null))) return null;
  try {
    const { stdout } = await execFileAsync(mlPython, [exifScript, file], { timeout: 15000, maxBuffer: 64 * 1024 });
    const value = stdout.trim();
    return value && /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : null;
  } catch {
    return null;
  }
}

async function savePhoto(payload) {
  if (!validFeatures(payload?.features)) throw Object.assign(new Error('Не удалось обработать изображение. Загрузите его снова.'), { status: 400 });
  const { bytes, extension } = decodeImage(payload.dataUrl);
  const capturedOn = optionalDate(payload.capturedOn, 'Проверьте дату съёмки.');
  const id = crypto.randomUUID();
  const file = `${id}.${extension}`;
  const target = path.join(imageDir, file);
  await fs.writeFile(target, bytes);
  const takenAt = await readExifDate(target);
  return { id, url: `/images/${file}`, features: payload.features, createdAt: new Date().toISOString(),
    takenAt, capturedOn, foal: payload.foal === true };
}

function similarity(a, b) {
  let differences = 0;
  const ah = BigInt(`0x${a.hash}`), bh = BigInt(`0x${b.hash}`);
  let xor = ah ^ bh;
  while (xor) { differences++; xor &= xor - 1n; }
  const hash = 1 - differences / 64;
  const histogram = a.histogram.reduce((sum, n, i) => sum + Math.min(n, b.histogram[i]), 0) / 3;
  const grid = 1 - a.grid.reduce((sum, n, i) => sum + Math.abs(n - b.grid[i]), 0) / 64;
  return Math.max(0, Math.min(1, 0.46 * hash + 0.30 * histogram + 0.24 * grid));
}

// Closed-set scores over the whole catalog become relative class
// probabilities. The catalog can hold any number of individuals, so the
// number of classes grows with the catalog without code changes.
function toClassification(entries, scoreOf) {
  if (!entries.length) return [];
  const values = entries.map(scoreOf);
  const max = Math.max(...values);
  return entries.map((entry, index) => ({
    horseId: entry.horse.id,
    name: entry.horse.name,
    similarity: Math.max(0, Math.round((values[index] / max) * 1000) / 1000)
  })).sort((a, b) => b.similarity - a.similarity);
}

async function ensureVision() {
  if (process.platform !== 'darwin') throw new Error('Vision requires macOS');
  if (visionBuild) return visionBuild;
  visionBuild = (async () => {
    const [source, binary] = await Promise.all([
      fs.stat(visionSource),
      fs.stat(visionBinary).catch(() => null)
    ]);
    if (binary && binary.mtimeMs >= source.mtimeMs) return visionBinary;
    const sdk = '/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk';
    const args = ['-O'];
    if (await fs.stat(sdk).catch(() => null)) args.push('-sdk', sdk);
    args.push(visionSource, '-o', visionBinary);
    await execFileAsync('swiftc', args, {
      timeout: 120000,
      maxBuffer: 1024 * 1024,
      env: { ...process.env, CLANG_MODULE_CACHE_PATH: path.join(tempDir, 'clang-cache'), SWIFT_MODULE_CACHE_PATH: path.join(tempDir, 'swift-cache') }
    });
    return visionBinary;
  })().catch(error => { visionBuild = null; throw error; });
  return visionBuild;
}

async function identifyWithVision(dataUrl, eligibleHorses) {
  const { bytes, extension } = decodeImage(dataUrl);
  const queryPath = path.join(tempDir, `${crypto.randomUUID()}.${extension}`);
  await fs.writeFile(queryPath, bytes);
  try {
    const binary = await ensureVision();
    const references = eligibleHorses.flatMap(horse => horse.photos.map(photo => ({ horse, photo,
      path: path.join(imageDir, path.basename(photo.url)) })));
    if (!references.length) return { candidates: [], classification: [] };
    const { stdout } = await execFileAsync(binary, [queryPath, ...references.map(reference => reference.path)], {
      timeout: 120000, maxBuffer: 4 * 1024 * 1024
    });
    const matches = JSON.parse(stdout);
    const bestByHorse = new Map();
    for (const match of matches) {
      const reference = references[match.index];
      if (!reference || typeof match.distance !== 'number' || !Number.isFinite(match.distance)) continue;
      const previous = bestByHorse.get(reference.horse.id);
      if (!previous || match.distance < previous.distance) {
        bestByHorse.set(reference.horse.id, { horse: publicHorse(reference.horse),
          distance: Math.round(match.distance * 1000) / 1000,
          photoUrl: reference.photo.url, region: match.region });
      }
    }
    const values = [...bestByHorse.values()];
    return { candidates: values.sort((a, b) => a.distance - b.distance).slice(0, 5),
      classification: toClassification(values, entry => 1 / (1 + entry.distance)) };
  } finally { await fs.unlink(queryPath).catch(() => {}); }
}

async function identifyWithMl(dataUrl, eligibleHorses) {
  const { bytes, extension } = decodeImage(dataUrl);
  const references = eligibleHorses.flatMap(horse => horse.photos.map(photo => ({ horse, photo,
    path: path.join(imageDir, path.basename(photo.url)) })));
  const queryPath = path.join(tempDir, `${crypto.randomUUID()}.${extension}`);
  await fs.writeFile(queryPath, bytes);
  try {
    const { stdout } = await execFileAsync(mlPython, [mlScript, queryPath, ...references.map(ref => ref.path)], {
      timeout: 120000, maxBuffer: 4 * 1024 * 1024,
      env: { ...process.env, HF_HOME: path.join(dataDir, 'hf'), HF_HUB_OFFLINE: '1', TAKHI_CPU: '1' }
    });
    const result = JSON.parse(stdout);
    const matches = result.matches;
    const byHorse = new Map();
    for (const match of matches) {
      const reference = references[match.index];
      if (!reference || typeof match.similarity !== 'number' || !Number.isFinite(match.similarity)) continue;
      if (!byHorse.has(reference.horse.id)) byHorse.set(reference.horse.id, []);
      byHorse.get(reference.horse.id).push({ horse: publicHorse(reference.horse),
        similarity: match.similarity, photoUrl: reference.photo.url });
    }
    // Mean of the two closest photos per horse is more stable than a single max.
    const values = [...byHorse.values()].map(entries => {
      const sorted = entries.sort((a, b) => b.similarity - a.similarity);
      const score = sorted.length > 1 ? (sorted[0].similarity + sorted[1].similarity) / 2 : sorted[0].similarity;
      return { horse: sorted[0].horse, similarity: Math.round(score * 1e5) / 1e5, photoUrl: sorted[0].photoUrl };
    });
    return { species: result.species,
      candidates: values.sort((a, b) => b.similarity - a.similarity).slice(0, 5),
      classification: toClassification(values, entry => entry.similarity) };
  } finally { await fs.unlink(queryPath).catch(() => {}); }
}

async function findHorseBoxes(bytes, extension) {
  const queryPath = path.join(tempDir, `${crypto.randomUUID()}.${extension}`);
  try {
    await fs.writeFile(queryPath, bytes);
    const args = [detectorScript, detectorWeights, queryPath];
    const { stdout } = await execFileAsync(mlPython, args, { timeout: 45000, maxBuffer: 64 * 1024 });
    return JSON.parse(stdout.trim().split('\n').at(-1));
  } catch (error) {
    console.error('Horse detector unavailable:', error.stderr || error.message);
    return null;
  } finally {
    await fs.unlink(queryPath).catch(() => {});
  }
}

async function detectHorses(dataUrl) {
  const { bytes, extension } = decodeImage(dataUrl);
  const [runtime, weights] = await Promise.all(
    [mlPython, detectorWeights].map(file => fs.stat(file).catch(() => null))
  );
  if (!runtime || !weights) return null;
  const key = crypto.createHash('sha256').update(`${weights.mtimeMs}:${weights.size}:`).update(bytes).digest('hex');
  const cached = detectorCache.get(key);
  if (cached) {
    detectorCache.delete(key);
    detectorCache.set(key, cached);
    return cached;
  }
  const pending = findHorseBoxes(bytes, extension);
  detectorCache.set(key, pending);
  if (detectorCache.size > detectorCacheLimit) detectorCache.delete(detectorCache.keys().next().value);
  const boxes = await pending;
  if (boxes === null && detectorCache.get(key) === pending) detectorCache.delete(key);
  return boxes;
}

async function segmentWithDino(dataUrl) {
  const { bytes, extension } = decodeImage(dataUrl);
  const file = path.join(tempDir, `${crypto.randomUUID()}.${extension}`);
  try {
    await fs.writeFile(file, bytes);
    const { stdout } = await execFileAsync(mlPython,
      [dinoPartsScript, 'predict', dinoPartsCheckpoint, '--image', file, '--weights', dinoPartsBackbone],
      { timeout: 90000, maxBuffer: 1024 * 1024, env: { ...process.env, HF_HUB_OFFLINE: '1' } });
    return JSON.parse(stdout);
  } finally {
    await fs.unlink(file).catch(() => {});
  }
}

const routeGrowth = growthRoutes({ dataDir, shareMode, send, readJson });

async function api(req, res, pathname) {
  if (await routeGrowth(req, res, pathname)) return;
  if (req.method === 'POST' && pathname === '/api/parts') {
    if (activeIdentifications >= maxActiveIdentifications) {
      return send(res, 429, { error: 'Модель занята. Повторите запрос через несколько секунд.' });
    }
    activeIdentifications++;
    try {
      const body = await readJson(req);
      const { bytes, extension } = decodeImage(body.dataUrl);
      if (!(await fs.stat(mlPython).catch(() => null)) || !(await fs.stat(partsScript).catch(() => null))) {
        return send(res, 200, { available: false });
      }
      const queryPath = path.join(tempDir, `${crypto.randomUUID()}.${extension}`);
      try {
        await fs.writeFile(queryPath, bytes);
        const { stdout } = await execFileAsync(mlPython, [partsScript, partsWeights, queryPath], {
          timeout: 90000, maxBuffer: 16 * 1024 * 1024
        });
        return send(res, 200, JSON.parse(stdout.trim().split('\n').at(-1)));
      } finally {
        await fs.unlink(queryPath).catch(() => {});
      }
    } catch (error) {
      console.error('Part segmentation unavailable:', error.stderr || error.message);
      return send(res, 200, { available: false });
    } finally {
      activeIdentifications--;
    }
  }
  if (req.method === 'POST' && pathname === '/api/segment-parts') {
    if (shareMode) return send(res, 404, { error: 'Маски доступны в локальном режиме.' });
    if (segmenting) return send(res, 429, { error: 'Модель частей тела занята.' });
    if (!(await fs.stat(dinoPartsCheckpoint).catch(() => null)) || !(await fs.stat(dinoPartsBackbone).catch(() => null))) {
      return send(res, 503, { error: 'Модель частей тела ждёт обучения на ручной разметке.' });
    }
    segmenting = true;
    try {
      const body = await readJson(req);
      return send(res, 200, await segmentWithDino(body.dataUrl));
    } catch (error) {
      console.error('Part segmentation unavailable:', error.stderr || error.message);
      return send(res, 503, { error: 'Не удалось запустить модель частей тела.' });
    } finally {
      segmenting = false;
    }
  }
  if (req.method === 'POST' && pathname === '/api/detect-horses') {
    if (activeIdentifications >= maxActiveIdentifications) {
      return send(res, 429, { error: 'Модель занята. Повторите запрос через несколько секунд.' });
    }
    activeIdentifications++;
    try {
      const boxes = await detectHorses((await readJson(req))?.dataUrl);
      return send(res, 200, { available: boxes !== null, boxes: boxes ?? [] });
    } finally {
      activeIdentifications--;
    }
  }
  if (req.method === 'GET' && pathname === '/api/config') {
    const partsReady = !!(await fs.stat(dinoPartsCheckpoint).catch(() => null)) && !!(await fs.stat(dinoPartsBackbone).catch(() => null));
    return send(res, 200, { canEditCatalog: !shareMode, partsAvailable: partsReady });
  }
  if (req.method === 'POST' && pathname === '/api/identity-reviews') {
    if (shareMode) return send(res, 403, { error: 'Сохранять разметку можно только в локальном режиме.' });
    const body = await readJson(req);
    const horse = horses.find(item => item.id === body?.horseId);
    if (!horse) return send(res, 404, { error: 'Выбранной особи нет в каталоге.' });
    const { bytes } = decodeImage(body.dataUrl);
    const id = crypto.createHash('sha256').update(bytes).digest('hex');
    const record = { id, horseId: horse.id, horseName: horse.name,
      suggestedHorseId: horses.find(item => item.id === body.suggestedHorseId)?.id ?? null,
      reviewedAt: new Date().toISOString(), dataUrl: body.dataUrl };
    await fs.mkdir(reviewDir, { recursive: true });
    const target = path.join(reviewDir, `${id}.json`);
    const temporary = `${target}.${crypto.randomUUID()}.tmp`;
    try {
      await fs.writeFile(temporary, JSON.stringify(record), { mode: 0o600 });
      await fs.rename(temporary, target);
    } finally {
      await fs.unlink(temporary).catch(error => { if (error.code !== 'ENOENT') throw error; });
    }
    return send(res, 200, { id, horseId: horse.id, horseName: horse.name });
  }
  if (shareMode && ((pathname === '/api/horses' && req.method === 'POST') ||
      (/^\/api\/horses\/[0-9a-f-]+(?:\/photos)?$/.test(pathname) && req.method !== 'GET'))) {
    return send(res, 403, { error: 'В общем доступе каталог открыт только для просмотра.' });
  }
  if (req.method === 'GET' && pathname === '/api/horses') return send(res, 200, { horses: horses.map(publicHorse) });
  if (req.method === 'POST' && pathname === '/api/horses') {
    const body = await readJson(req);
    const name = String(body.name || '').trim();
    const notes = String(body.notes || '').trim();
    if (!name || name.length > 80 || notes.length > 500) return send(res, 400, { error: 'Укажите кличку или полевой номер (до 80 символов).' });
    const photos = body.photo ? [await savePhoto(body.photo)] : [];
    const horse = { id: crypto.randomUUID(), name, notes, createdAt: new Date().toISOString(), photos };
    horses.unshift(horse);
    await persist();
    return send(res, 201, { horse: publicHorse(horse) });
  }
  const horseMatch = /^\/api\/horses\/([0-9a-f-]+)$/.exec(pathname);
  if (req.method === 'PATCH' && horseMatch) {
    const body = await readJson(req);
    const horse = horses.find(item => item.id === horseMatch[1]);
    if (!horse) return send(res, 404, { error: 'Особь не найдена.' });
    if (!body || typeof body !== 'object' || Array.isArray(body)) return send(res, 400, { error: 'Некорректная родословная.' });
    editPedigree(horses, horse, { ...horse, ...body });
    await persist();
    return send(res, 200, { horses: horses.map(publicHorse) });
  }
  const photoMatch = /^\/api\/horses\/([0-9a-f-]+)\/photos$/.exec(pathname);
  if (req.method === 'POST' && photoMatch) {
    const horse = horses.find(item => item.id === photoMatch[1]);
    if (!horse) return send(res, 404, { error: 'Особь не найдена.' });
    const body = await readJson(req);
    const photo = await savePhoto(body.photo);
    horse.photos.push(photo);
    await persist();
    return send(res, 201, { horse: publicHorse(horse) });
  }
  if (req.method === 'POST' && pathname === '/api/identify') {
    if (activeIdentifications >= maxActiveIdentifications) return send(res, 429, { error: 'Модель занята. Повторите запрос через несколько секунд.' });
    activeIdentifications++;
    try {
      const body = await readJson(req);
      if (!validFeatures(body.features)) return send(res, 400, { error: 'Не удалось обработать изображение.' });
      const capturedOn = optionalDate(body.capturedOn, 'Проверьте дату съёмки.');
      const eligibleHorses = horses.filter(horse => couldBeAlive(horse, capturedOn));
      const excludedByDate = horses.length - eligibleHorses.length;
      const boxes = await detectHorses(body.dataUrl);
      if (boxes === null) return send(res, 503, { error: 'Детектор лошадей временно недоступен. Попробуйте позже.' });
      if (!boxes.length) {
        return send(res, 200, { method: 'detector', species: { decision: 'no_horse_detected' }, candidates: [] });
      }
      if (body.dataUrl && await fs.stat(mlPython).catch(() => null)) {
        try {
          const result = await identifyWithMl(body.dataUrl, eligibleHorses);
          const trained = !!(await fs.stat(mlHead).catch(() => null));
          return send(res, 200, { method: trained ? 'trained' : 'megadescriptor', excludedByDate, ...result });
        } catch (error) {
          console.error('MegaDescriptor unavailable, using Vision:', error.stderr || error.message);
        }
      }
      if (body.dataUrl && process.platform === 'darwin') {
        try {
          const result = await identifyWithVision(body.dataUrl, eligibleHorses);
          return send(res, 200, { method: 'vision', excludedByDate, ...result });
        } catch (error) {
          console.error('Vision unavailable, using visual features:', error.message);
        }
      }
      const ranked = eligibleHorses.filter(horse => horse.photos.length).map(horse => {
        const best = horse.photos.reduce((current, photo) => {
          const score = similarity(body.features, photo.features);
          return score > current.score ? { score, photoId: photo.id, photoUrl: photo.url } : current;
        }, { score: -1 });
        return { horse: publicHorse(horse), score: Math.round(best.score * 100), photoUrl: best.photoUrl };
      }).sort((a, b) => b.score - a.score);
      return send(res, 200, { method: 'visual', excludedByDate, candidates: ranked.slice(0, 5),
        classification: toClassification(ranked, entry => entry.score / 100) });
    } finally { activeIdentifications--; }
  }
  if (req.method === 'DELETE' && /^\/api\/horses\/[0-9a-f-]+$/.test(pathname)) {
    const id = pathname.split('/')[3];
    const index = horses.findIndex(item => item.id === id);
    if (index < 0) return send(res, 404, { error: 'Особь не найдена.' });
    if (horses.some(horse => horse.motherId === id || horse.fatherId === id || horse.partnerIds?.includes(id))) {
      return send(res, 409, { error: 'На особь ссылаются другие карточки. Сначала уберите эти связи.' });
    }
    const [removed] = horses.splice(index, 1);
    await persist();
    const files = removed.photos.flatMap(photo => {
      const name = path.basename(photo.url);
      return [path.join(imageDir, name), path.join(referenceCacheDir, `${name}.pt`)];
    });
    await Promise.allSettled(files.map(file => fs.unlink(file)));
    return send(res, 200, { ok: true });
  }
  return send(res, 404, { error: 'Маршрут не найден.' });
}

const server = http.createServer(async (req, res) => {
  try {
    if (!hasShareAccess(req.headers.authorization)) {
      res.writeHead(401, { 'WWW-Authenticate': 'Basic realm="Takhi", charset="UTF-8"',
        'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
      res.end('Требуется пароль для доступа к Тахи.');
      return;
    }
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (pathname.startsWith('/api/')) return await api(req, res, pathname);
    if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, { error: 'Метод не поддерживается.' });
    const file = pathname === '/' ? path.join(publicDir, 'index.html') : pathname.startsWith('/images/')
      ? path.join(imageDir, path.basename(pathname)) : path.join(publicDir, pathname);
    if (!file.startsWith(publicDir + path.sep) && !file.startsWith(imageDir + path.sep)) return send(res, 404, { error: 'Файл не найден.' });
    const body = await fs.readFile(file);
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream',
      'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store' });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch (error) {
    if (error.code === 'ENOENT') return send(res, 404, { error: 'Файл не найден.' });
    if (!error.status) console.error(error);
    send(res, error.status || 500, { error: error.status ? error.message : 'Внутренняя ошибка сервера.' });
  }
});

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  server.listen(port, host, () => console.log(`Takhi Fieldbook: http://${host}:${port}`));
}
export { server, similarity, validFeatures, toClassification };
