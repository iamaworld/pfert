const section = document.querySelector('#growth');
const get = id => section.querySelector(`#growth-${id}`);
const horseSelect = get('horse');
const partSelect = get('part');
const editor = get('editor');
const dateFormat = new Intl.DateTimeFormat('ru', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
const dayCount = new Intl.NumberFormat('ru');
const colors = { body: '#55ccbd', legs: '#ffa85e', ears: '#bcadff' };
const parts = Object.keys(colors);
let archive = [];
let catalog = [];
let diaries = [];
let manual = null;
let early = 0;
let late = 0;

const formatDate = day => dateFormat.format(new Date(day));
const makeOption = (text, value) => new Option(text, value);

export function drawPartMasks(canvas, polygons, show, draft = [], draftPart = 'ears') {
  const context = canvas.getContext('2d');
  canvas.width = 1000;
  canvas.height = Math.max(1, Math.round(1000 * canvas.clientHeight / (canvas.clientWidth || 1000)));
  const trace = points => {
    context.beginPath();
    points.forEach(([x, y], index) => context[index ? 'lineTo' : 'moveTo'](x * canvas.width, y * canvas.height));
  };
  for (const part of parts) {
    if (show !== 'all' && show !== part) continue;
    for (const polygon of polygons?.[part] ?? []) {
      trace(polygon);
      context.closePath();
      context.fillStyle = colors[part] + '59';
      context.strokeStyle = colors[part];
      context.lineWidth = 3;
      context.fill();
      context.stroke();
    }
  }
  if (!draft.length) return;
  trace(draft);
  context.strokeStyle = colors[draftPart];
  context.lineWidth = 3;
  context.stroke();
  context.fillStyle = colors[draftPart];
  for (const [x, y] of draft) {
    context.beginPath();
    context.arc(x * canvas.width, y * canvas.height, 4, 0, Math.PI * 2);
    context.fill();
  }
}

function picture(photo, name, show) {
  const stage = document.createElement('div');
  stage.className = 'growth-image';
  const image = new Image();
  image.alt = `${name}, ${formatDate(photo.capturedOn)}`;
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  stage.append(image, canvas);
  image.onload = () => drawPartMasks(canvas, photo.annotation?.polygons, show);
  image.src = photo.url;
  return { stage, canvas, image };
}

function renderFrame(target, diary, selected, previous) {
  const photo = diary.photos[selected];
  target.replaceChildren();
  const label = document.createElement('label');
  label.textContent = previous ? 'Ранний снимок ' : 'Поздний снимок ';
  const dates = document.createElement('select');
  diary.photos.forEach((row, index) => dates.add(makeOption(formatDate(row.capturedOn), index)));
  dates.value = selected;
  dates.addEventListener('change', () => {
    if (previous) early = Number(dates.value);
    else late = Number(dates.value);
    if (early > late) [early, late] = [late, early];
    closeEditor();
    renderFrames();
  });
  label.append(dates);
  const { stage } = picture(photo, diary.name, partSelect.value);
  const caption = document.createElement('p');
  caption.className = 'growth-caption';
  caption.textContent = photo.annotation?.source === 'manual' ? 'Ручная разметка'
    : photo.annotation?.source === 'model' ? 'Маски модели' : 'Разметки пока нет';
  target.append(label, stage, caption);
  if (diary.fromArchive) {
    const annotate = document.createElement('button');
    annotate.type = 'button';
    annotate.className = 'subtle-button';
    annotate.textContent = 'Разметить части тела';
    annotate.addEventListener('click', () => openEditor(photo, diary.name));
    caption.append(' · ', annotate);
  }
}

function renderFrames() {
  const diary = diaries[Number(horseSelect.value)];
  get('controls').hidden = !diary;
  get('empty').hidden = !!diary;
  horseSelect.disabled = !diary;
  if (!diary) return;
  const elapsed = Math.round((Date.parse(diary.photos[late].capturedOn) - Date.parse(diary.photos[early].capturedOn)) / 86400000);
  get('interval').textContent = elapsed ? `Между снимками ${dayCount.format(elapsed)} дн.` : 'Одна дата съёмки';
  renderFrame(get('first'), diary, early, true);
  renderFrame(get('second'), diary, late, false);
}

function resetDates() {
  early = 0;
  late = Math.max(0, (diaries[Number(horseSelect.value)]?.photos.length ?? 1) - 1);
  closeEditor();
  renderFrames();
}

function renderDiaries() {
  const before = horseSelect.selectedOptions[0]?.textContent;
  const existing = catalog.map(horse => ({
    name: horse.name,
    photos: horse.photos.filter(photo => photo.capturedOn).sort((a, b) => a.capturedOn.localeCompare(b.capturedOn))
  })).filter(horse => horse.photos.length);
  diaries = [...archive, ...existing];
  horseSelect.replaceChildren();
  diaries.forEach((horse, index) => horseSelect.add(makeOption(horse.fromArchive ? `${horse.name} · архив` : horse.name, index)));
  const previous = [...horseSelect.options].find(option => option.textContent === before);
  if (previous) horseSelect.value = previous.value;
  resetDates();
}

function refreshEditor() {
  if (!manual) return;
  drawPartMasks(manual.canvas, manual.polygons, 'all', manual.draft, get('edit-part').value);
}

function openEditor(photo, name) {
  const { stage, image, canvas } = picture(photo, name, 'all');
  editor.hidden = false;
  const art = get('edit-image');
  art.replaceChildren(stage);
  manual = { photo, canvas, draft: [], polygons: structuredClone(photo.annotation?.polygons ?? {}) };
  image.onload = refreshEditor;
  get('note').textContent = photo.annotation?.source === 'model' ? 'Проверьте все контуры модели перед сохранением.' : '';
  const addPoint = (x, y) => {
    if (!manual || manual.draft.length >= 300) return;
    manual.draft.push([Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))]);
    refreshEditor();
  };
  canvas.removeAttribute('aria-hidden');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Добавьте точку контура кликом. Клавиши со стрелками перемещают курсор, Enter добавляет точку.');
  canvas.addEventListener('click', event => {
    const box = canvas.getBoundingClientRect();
    addPoint((event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height);
  });
  let cursor = [0.5, 0.5];
  canvas.addEventListener('keydown', event => {
    const movement = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
    if (!movement && event.key !== 'Enter') return;
    event.preventDefault();
    if (!movement) return addPoint(...cursor);
    cursor = cursor.map((coordinate, i) => Math.min(1, Math.max(0, coordinate + movement[i] * (event.shiftKey ? 0.04 : 0.01))));
    refreshEditor();
    const context = canvas.getContext('2d');
    context.fillStyle = '#fff';
    context.fillRect(cursor[0] * canvas.width - 4, cursor[1] * canvas.height - 4, 8, 8);
  });
  canvas.focus({ preventScroll: true });
  editor.scrollIntoView({ block: 'start' });
}

function closeEditor() {
  manual = null;
  editor.hidden = true;
}

get('edit-part').addEventListener('change', () => {
  if (!manual) return;
  manual.draft = [];
  refreshEditor();
});
get('close-polygon').addEventListener('click', () => {
  if (!manual || manual.draft.length < 3) {
    get('note').textContent = 'У контура должно быть хотя бы три точки.';
    return;
  }
  const part = get('edit-part').value;
  (manual.polygons[part] ??= []).push(manual.draft);
  manual.draft = [];
  get('note').textContent = 'Контур добавлен. Можно обвести следующую часть.';
  refreshEditor();
});
get('undo').addEventListener('click', () => {
  if (!manual) return;
  if (manual.draft.length) manual.draft.pop();
  else manual.polygons[get('edit-part').value]?.pop();
  refreshEditor();
});
get('cancel').addEventListener('click', closeEditor);
get('save').addEventListener('click', async event => {
  if (!manual) return;
  if (manual.draft.length) {
    get('note').textContent = 'Замкните текущий контур перед сохранением.';
    return;
  }
  const current = manual;
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const response = await fetch(`/api/growth/labels/${current.photo.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ polygons: current.polygons })
    });
    const record = await response.json();
    if (!response.ok) throw new Error(record.error);
    current.photo.annotation = record;
    if (manual === current) get('note').textContent = 'Ручная разметка сохранена.';
    renderFrames();
  } catch (error) {
    if (manual === current) get('note').textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

horseSelect.addEventListener('change', resetDates);
partSelect.addEventListener('change', renderFrames);
document.addEventListener('catalog-updated', event => {
  catalog = event.detail;
  renderDiaries();
});

try {
  const response = await fetch('/api/growth');
  if (response.ok) {
    const data = await response.json();
    archive = data.horses.filter(horse => horse.photos.length).map(horse => ({ ...horse, fromArchive: true }));
    get('source').textContent = data.partialArchive ? 'Архив загружен частично. Даты взяты из метаданных снимков.' : '';
    renderDiaries();
    section.hidden = false;
    document.querySelector('#growth-link').hidden = false;
  }
} catch {
  section.hidden = true;
}
