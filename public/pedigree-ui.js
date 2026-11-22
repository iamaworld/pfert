const dateFormat = new Intl.DateTimeFormat('ru', { dateStyle: 'medium', timeZone: 'UTC' });
const format = date => dateFormat.format(new Date(date));

export function createPedigreeDialog(state, request, updateCatalog) {
  const dialog = document.querySelector('#pedigree-dialog');
  const find = selector => dialog.querySelector(selector);
  const form = find('form');
  const field = name => form.elements.namedItem(name);
  const gallery = find('#pedigree-gallery');
  const status = find('#pedigree-status');
  let currentId = null;

  function peopleRow(label, horses) {
    const row = document.createElement('div');
    const title = document.createElement('dt');
    title.textContent = label;
    const value = document.createElement('dd');
    value.className = 'family-links';
    for (const horse of horses.filter(Boolean)) {
      const link = document.createElement('button');
      link.type = 'button';
      link.className = 'subtle-button';
      link.textContent = horse.name;
      link.addEventListener('click', () => open(horse.id));
      value.append(link);
    }
    if (!value.childElementCount) value.textContent = 'Нет сведений';
    row.append(title, value);
    return row;
  }

  function open(id) {
    const byId = new Map(state.horses.map(item => [item.id, item]));
    const horse = byId.get(id);
    if (!horse) return;
    currentId = id;
    find('#pedigree-title').textContent = horse.name;
    const dates = [];
    if (horse.sex) dates.push(horse.sex === 'female' ? 'Кобыла' : 'Жеребец');
    if (horse.birthOn) dates.push(`Рождение ${format(horse.birthOn)}`);
    if (horse.deceased) dates.push(horse.deathOn ? `Смерть ${format(horse.deathOn)}` : 'В мемориале');
    find('#pedigree-dates').textContent = dates.join(' · ');
    const children = state.horses.filter(item => item.motherId === id || item.fatherId === id);
    const common = new Set(children.map(item => item.motherId === id ? item.fatherId : item.motherId).filter(Boolean));
    const relatives = document.createElement('dl');
    relatives.className = 'family-grid';
    relatives.append(peopleRow('Мать', [byId.get(horse.motherId)]), peopleRow('Отец', [byId.get(horse.fatherId)]),
      peopleRow('Сыновья', children.filter(child => child.sex === 'male')),
      peopleRow('Дочери', children.filter(child => child.sex === 'female')),
      peopleRow('Потомство, пол неизвестен', children.filter(child => !child.sex)),
      peopleRow('Зафиксированные пары', horse.partnerIds.map(partner => byId.get(partner))),
      peopleRow('Особи с общим потомством', [...common].map(partner => byId.get(partner))));
    find('#pedigree-family').replaceChildren(relatives);

    gallery.replaceChildren();
    for (const photo of horse.photos.filter(photo => photo.foal)) {
      const figure = document.createElement('figure');
      const link = document.createElement('a');
      link.href = photo.url;
      link.target = '_blank';
      link.rel = 'noopener';
      const image = new Image();
      image.src = photo.url;
      image.alt = `${horse.name} в детстве`;
      image.loading = 'lazy';
      const caption = document.createElement('figcaption');
      caption.textContent = photo.capturedOn ? format(photo.capturedOn) : 'Дата съёмки неизвестна';
      link.append(image);
      figure.append(link, caption);
      gallery.append(figure);
    }
    if (!gallery.childElementCount) gallery.textContent = 'Фотографии жеребёнка пока не добавлены.';
    const editor = find('#pedigree-edit');
    editor.hidden = !state.canEditCatalog;
    editor.open = false;
    status.textContent = '';
    field('sex').value = horse.sex ?? '';
    field('birthOn').value = horse.birthOn ?? '';
    field('deathOn').value = horse.deathOn ?? '';
    field('status').value = horse.deceased ? 'deceased' : 'catalog';
    field('deathOn').disabled = !horse.deceased;
    const others = state.horses.filter(item => item.id !== id);
    for (const [name, wrongSex] of [['motherId', 'male'], ['fatherId', 'female']]) {
      const select = field(name);
      select.replaceChildren(new Option('Не указано', ''));
      for (const item of others.filter(other => other.sex !== wrongSex)) select.add(new Option(item.name, item.id));
      select.value = horse[name] ?? '';
    }
    find('#pedigree-partners').replaceChildren(...others.map(item => {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.name = 'partner';
      checkbox.value = item.id;
      checkbox.checked = horse.partnerIds.includes(item.id);
      label.append(checkbox, item.name);
      return label;
    }));
    if (!others.length) find('#pedigree-partners').textContent = 'Других особей в каталоге пока нет.';
    if (!dialog.open) dialog.showModal();
  }

  field('status').addEventListener('change', () => { field('deathOn').disabled = field('status').value !== 'deceased'; });
  find('#pedigree-close').addEventListener('click', () => dialog.close());
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const id = currentId;
    const button = form.querySelector('[type="submit"]');
    const data = new FormData(form);
    const body = {
      sex: data.get('sex'), birthOn: data.get('birthOn'), deathOn: data.get('deathOn'),
      motherId: data.get('motherId'), fatherId: data.get('fatherId'),
      deceased: data.get('status') === 'deceased', partnerIds: data.getAll('partner')
    };
    button.disabled = true;
    try {
      const result = await request(`/api/horses/${id}`, { method: 'PATCH', body: JSON.stringify(body) });
      state.horses = result.horses;
      updateCatalog();
      if (dialog.open && currentId === id) {
        open(id);
        find('#pedigree-edit').open = true;
        status.textContent = 'Сохранено.';
      }
    } catch (error) {
      if (dialog.open && currentId === id) status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
  return open;
}
