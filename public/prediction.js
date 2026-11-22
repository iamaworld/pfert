const scoreKey = 'takhi-prediction-points';
const fireworkBeatMs = 145;
const celebrationColors = ['#eebc61', '#fc8f85', '#76c6a2', '#c18bdd', '#5ca5d1'];
const species = [
  ['przewalski', 'Лошадь Пржевальского'],
  ['not_przewalski', 'Другая лошадь'],
  ['no_horse_detected', 'Лошадь не обнаружится']
];

function storedScore() {
  try {
    const value = Number(localStorage.getItem(scoreKey));
    return Number.isSafeInteger(value) && value >= 0 ? value : 0;
  } catch { return 0; }
}

function launchFireworks(panel) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const fireworks = document.createElement('div');
  fireworks.className = 'prediction-fireworks';
  fireworks.setAttribute('aria-hidden', 'true');
  for (let index = 0; index < 3; index++) {
    const burst = document.createElement('div');
    burst.className = 'prediction-burst';
    burst.style.setProperty('--origin', `${22 + index * 28}%`);
    burst.style.setProperty('--height', `${[35, 22, 45][index]}%`);
    burst.style.setProperty('--delay', `${index * fireworkBeatMs}ms`);
    burst.style.setProperty('--firework-color', celebrationColors[index]);
    for (let ray = 0; ray < 14; ray++) {
      const spark = document.createElement('i');
      const angle = Math.PI * 2 * ray / 14;
      const radius = 42 + (ray % 3) * 18;
      spark.style.setProperty('--dx', `${Math.cos(angle) * radius}px`);
      spark.style.setProperty('--dy', `${Math.sin(angle) * radius}px`);
      burst.append(spark);
    }
    fireworks.append(burst);
  }
  for (let index = 0; index < 36; index++) {
    const piece = document.createElement('b');
    const lane = index % 12;
    const wave = Math.floor(index / 12);
    piece.className = 'prediction-confetti';
    piece.style.setProperty('--left', `${5 + lane * 8}%`);
    piece.style.setProperty('--confetti-color', celebrationColors[(index + wave) % celebrationColors.length]);
    piece.style.setProperty('--delay', `${wave * 130 + (lane % 4) * 25}ms`);
    piece.style.setProperty('--drift', `${(lane - 5.5) * (13 + wave * 3)}px`);
    piece.style.setProperty('--twist', `${(lane % 2 ? -1 : 1) * (240 + lane * 35)}deg`);
    fireworks.append(piece);
  }
  panel.append(fireworks);
  setTimeout(() => fireworks.remove(), 1900);
}

export function createPredictionGame(section, resultPanel) {
  const form = section.querySelector('form');
  const select = form.querySelector('select');
  const feedback = section.querySelector('#prediction-feedback');
  const score = section.querySelector('#prediction-score');
  let points = storedScore();
  let active = false;
  let pick = '';
  let settled = true;

  const renderScore = () => { score.textContent = `Очки: ${points}`; };
  const lock = () => {
    active = false;
    form.hidden = true;
  };

  form.addEventListener('submit', event => {
    event.preventDefault();
    if (!active || !select.value) return;
    pick = select.value;
    feedback.textContent = `Прогноз принят: ${select.selectedOptions[0].textContent}.`;
    lock();
  });

  function start(horses) {
    active = true;
    pick = '';
    settled = false;
    feedback.textContent = '';
    delete section.dataset.result;
    select.replaceChildren(new Option('Выберите вариант', ''));
    const known = horses.filter(horse => horse.photos.length);
    if (known.length) {
      const group = document.createElement('optgroup');
      group.label = 'Особи';
      group.append(...known.map(horse => new Option(horse.name, `horse:${horse.id}`)));
      select.append(group);
    }
    const kinds = document.createElement('optgroup');
    kinds.label = 'Вид на снимке';
    kinds.append(...species.map(([key, label]) => new Option(label, `species:${key}`)));
    select.append(kinds);
    renderScore();
    section.hidden = false;
    form.hidden = false;
  }

  function finish(result) {
    if (settled) return;
    settled = true;
    lock();
    if (!pick) {
      section.hidden = true;
      return;
    }
    const decision = result.species?.decision;
    const candidate = ['no_horse_detected', 'not_przewalski'].includes(decision)
      ? null : result.candidates[0]?.horse;
    const bySpecies = pick.startsWith('species:');
    const answer = bySpecies
      ? species.some(([key]) => key === decision) && `species:${decision}`
      : candidate && `horse:${candidate.id}`;
    if (!answer) {
      feedback.textContent = 'Модель не дала ответа для этого прогноза.';
    } else if (pick === answer) {
      if (points < Number.MAX_SAFE_INTEGER) points++;
      try { localStorage.setItem(scoreKey, String(points)); } catch { /* In-memory score still works. */ }
      renderScore();
      section.dataset.result = 'win';
      feedback.textContent = 'Прогноз совпал с ответом модели. +1 очко!';
      launchFireworks(resultPanel);
      resultPanel.dispatchEvent(new CustomEvent('prediction:win', { detail: { beatMs: fireworkBeatMs } }));
    } else {
      feedback.textContent = 'Прогноз не совпал с ответом модели.';
    }
  }

  function cancel() {
    settled = true;
    lock();
    section.hidden = true;
  }

  return { start, lock, finish, cancel };
}
