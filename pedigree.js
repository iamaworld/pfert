const invalid = message => Object.assign(new Error(message), { status: 400 });

export function optionalDate(value, message = 'Проверьте дату.') {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value) ||
      Number.isNaN(Date.parse(value)) || new Date(value).toISOString().slice(0, 10) !== value) throw invalid(message);
  return value;
}

export function editPedigree(horses, horse, changes) {
  const byId = new Map(horses.map(item => [item.id, item]));
  const link = value => {
    if (value === undefined || value === null || value === '') return null;
    if (value === horse.id || !byId.has(value)) throw invalid('Выберите другую существующую особь.');
    return value;
  };
  const birthOn = optionalDate(changes.birthOn, 'Проверьте дату рождения.');
  const deceased = changes.deceased === true;
  const deathOn = deceased ? optionalDate(changes.deathOn, 'Проверьте дату смерти.') : null;
  if (birthOn && deathOn && birthOn > deathOn) throw invalid('Дата смерти раньше даты рождения.');
  const sex = changes.sex || null;
  if (sex !== null && sex !== 'female' && sex !== 'male') throw invalid('Проверьте пол особи.');
  const motherId = link(changes.motherId);
  const fatherId = link(changes.fatherId);
  if (motherId && motherId === fatherId) throw invalid('Мать и отец должны быть разными особями.');
  if (byId.get(motherId)?.sex === 'male' || byId.get(fatherId)?.sex === 'female' ||
      (sex === 'male' && horses.some(item => item.motherId === horse.id)) ||
      (sex === 'female' && horses.some(item => item.fatherId === horse.id))) {
    throw invalid('Указанный пол противоречит роли в родословной.');
  }
  const pending = [motherId, fatherId].filter(Boolean);
  const seen = new Set();
  while (pending.length) {
    const parent = pending.pop();
    if (parent === horse.id) throw invalid('Эта связь создаёт цикл в родословной.');
    if (seen.has(parent)) continue;
    seen.add(parent);
    const row = byId.get(parent);
    if (row?.motherId) pending.push(row.motherId);
    if (row?.fatherId) pending.push(row.fatherId);
  }
  const submitted = changes.partnerIds ?? [];
  if (!Array.isArray(submitted) || submitted.length > horses.length) throw invalid('Проверьте выбранные пары.');
  const partnerIds = [...new Set(submitted.map(link))].filter(Boolean);
  for (const other of horses) {
    if (other === horse) continue;
    const current = other.partnerIds ?? [];
    if (partnerIds.includes(other.id) && !current.includes(horse.id)) other.partnerIds = [...current, horse.id];
    else if (!partnerIds.includes(other.id) && current.includes(horse.id)) {
      other.partnerIds = current.filter(id => id !== horse.id);
    }
  }
  Object.assign(horse, { birthOn, deceased, deathOn, sex, motherId, fatherId, partnerIds });
}

export function couldBeAlive(horse, capturedOn) {
  return !capturedOn || ((!horse.birthOn || horse.birthOn <= capturedOn) &&
    (!horse.deceased || !horse.deathOn || horse.deathOn >= capturedOn));
}
