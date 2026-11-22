// Prepare the Kaggle part-segmentation dataset from the local catalog.
//
// Picks ~50 date-diverse photos (3 per horse + extras for the widest date
// spans), copies them to kaggle/dataset/images/ and writes a starter
// labels.json that tools/labeler.html will fill in.
//
// Existing user labels are never overwritten.

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const catalogFile = path.join(root, 'data', 'horses.json');
const imageDir = path.join(root, 'data', 'images');
const outDir = path.join(root, 'kaggle', 'dataset');
const outImages = path.join(outDir, 'images');
const labelsFile = path.join(outDir, 'labels.json');
const metadataFile = path.join(outDir, 'dataset-metadata.json');

const horses = JSON.parse(await fs.readFile(catalogFile, 'utf8'));
await fs.mkdir(outImages, { recursive: true });

function sortedPhotos(horse) {
  return [...horse.photos].sort((a, b) => String(a.takenAt || '').localeCompare(String(b.takenAt || '')));
}

function pick(photos, count) {
  if (photos.length <= count) return photos;
  const indexes = new Set([...Array(count)].map((_, i) => Math.round(i * (photos.length - 1) / (count - 1))));
  return photos.filter((_, i) => indexes.has(i));
}

const perHorse = 3;
const extraCount = 5;
const chosen = [];
for (const horse of horses) {
  const photos = sortedPhotos(horse);
  chosen.push(...pick(photos, perHorse).map(photo => ({ horse: horse.name, photo })));
}
const spans = horses
  .map(horse => ({ horse: horse.name, span: sortedPhotos(horse) }))
  .filter(item => item.span.length > perHorse)
  .sort((a, b) => b.span.length - a.span.length)
  .slice(0, extraCount);
for (const item of spans) {
  const extra = sortedPhotos(horses.find(horse => horse.name === item.horse)).find(photo => !chosen.some(entry => entry.photo.url === photo.url));
  if (extra) chosen.push({ horse: item.horse, photo: extra });
}

let copied = 0;
for (const { horse, photo } of chosen) {
  const source = path.join(imageDir, path.basename(photo.url));
  const target = path.join(outImages, `${horse}__${path.basename(photo.url)}`);
  await fs.copyFile(source, target);
  copied++;
}

let labels = {};
try {
  const existing = JSON.parse(await fs.readFile(labelsFile, 'utf8'));
  if (existing && Object.keys(existing).length) labels = existing;
} catch {}

await fs.writeFile(labelsFile, JSON.stringify(labels, null, 2) + '\n');
await fs.writeFile(metadataFile, JSON.stringify({
  title: 'takhi-parts',
  id: 'iamapeace/takhi-parts',
  licenses: [{ name: 'CC0-1.0' }]
}, null, 2) + '\n');

console.log(`Kaggle dataset prepared: ${copied} photos -> ${outDir}`);
console.log('Next: open tools/labeler.html, load kaggle/dataset/images, draw ears/legs/body polygons,');
console.log('download labels.json and replace kaggle/dataset/labels.json with it.');
