import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const idPattern = /^[a-f0-9]{20}$/;
const partNames = ['body', 'legs', 'ears'];

async function readJsonFile(file, fallback) {
  try { return JSON.parse(await fs.readFile(file, 'utf8')); }
  catch (error) {
    if (error.code === 'ENOENT') return fallback;
    throw error;
  }
}

function normalizePolygons(value) {
  if (!value || typeof value !== 'object') return null;
  const parts = {};
  for (const part of partNames) {
    const polygons = value[part] ?? [];
    if (!Array.isArray(polygons) || polygons.length > 20) return null;
    for (const polygon of polygons) {
      if (!Array.isArray(polygon) || polygon.length < 3 || polygon.length > 300) return null;
      for (const point of polygon) {
        if (!Array.isArray(point) || point.length !== 2 ||
            !point.every(coordinate => Number.isFinite(coordinate) && coordinate >= 0 && coordinate <= 1)) return null;
      }
    }
    parts[part] = polygons;
  }
  return partNames.some(part => parts[part].length) ? parts : null;
}

export function growthRoutes({ dataDir, shareMode, send, readJson }) {
  const directory = path.join(dataDir, 'growth');

  return async function route(req, res, pathname) {
    if (!pathname.startsWith('/api/growth')) return false;
    if (shareMode) {
      send(res, 404, { error: 'Дневник архива доступен локально.' });
      return true;
    }
    const manifest = await readJsonFile(path.join(directory, 'manifest.json'), { horses: {} });
    if (pathname === '/api/growth' && req.method === 'GET') {
      const predictions = await readJsonFile(path.join(directory, 'predictions.json'), {});
      const horses = await Promise.all(Object.entries(manifest.horses).map(async ([name, photos]) => ({
        name,
        photos: await Promise.all(photos.filter(photo => idPattern.test(photo.id)).map(async photo => ({
          ...photo, url: `/api/growth/photos/${photo.id}`,
          annotation: await readJsonFile(path.join(directory, 'labels', `${photo.id}.json`), null) ?? predictions[photo.id] ?? null
        })))
      })));
      send(res, 200, { horses, partialArchive: !!manifest.partialArchive });
      return true;
    }

    const match = /^\/api\/growth\/(photos|labels)\/([a-f0-9]{20})$/.exec(pathname);
    const exists = match && Object.values(manifest.horses).some(photos => photos.some(photo => photo.id === match[2]));
    if (!exists) {
      send(res, 404, { error: 'Наблюдение не найдено.' });
      return true;
    }
    const [, resource, id] = match;
    if (resource === 'photos' && req.method === 'GET') {
      send(res, 200, await fs.readFile(path.join(directory, 'photos', `${id}.jpg`)), 'image/jpeg');
    } else if (resource === 'labels' && req.method === 'PUT') {
      const polygons = normalizePolygons((await readJson(req))?.polygons);
      if (!polygons) {
        send(res, 400, { error: 'Добавьте корректный контур хотя бы одной части тела.' });
        return true;
      }
      const record = { source: 'manual', polygons };
      const labelsDir = path.join(directory, 'labels');
      await fs.mkdir(labelsDir, { recursive: true });
      const target = path.join(labelsDir, `${id}.json`);
      const temporary = `${target}.${crypto.randomUUID()}.tmp`;
      try {
        await fs.writeFile(temporary, JSON.stringify(record));
        await fs.rename(temporary, target);
      } finally {
        await fs.unlink(temporary).catch(error => { if (error.code !== 'ENOENT') throw error; });
      }
      send(res, 200, record);
    } else {
      send(res, 405, { error: 'Метод не поддерживается.' });
    }
    return true;
  };
}
