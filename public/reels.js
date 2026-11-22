export const horseSymbol = '♞';
const symbols = [horseSymbol, '♦', '✦', '7', '♣', '♥'];
const resting = [0, 1, 2];
const minSpinMs = 4200;
const burstDurationMs = 900;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

function makeMachine() {
  const machine = document.querySelector('#reels-template').content.firstElementChild.cloneNode(true);
  const windows = machine.querySelector('.slot-reels');
  resting.forEach((offset, index) => {
    const reel = document.createElement('div');
    reel.className = 'slot-reel';
    const track = document.createElement('div');
    track.className = 'slot-track';
    track.style.setProperty('--speed', `${[590, 690, 630][index]}ms`);
    for (let row = 0; row < symbols.length * 3; row++) {
      const symbol = document.createElement('span');
      symbol.textContent = symbols[(row + offset) % symbols.length];
      track.append(symbol);
    }
    reel.append(track);
    windows.append(reel);
  });
  return machine;
}

export function showReels(panel, title, description, rolling = true, restart = false) {
  let machine = panel.querySelector('.slot-machine');
  if (!machine || ['settling', 'triple'].includes(machine.dataset.phase)) {
    machine = makeMachine();
    panel.replaceChildren(machine);
  }
  if (rolling && (restart || machine.dataset.phase !== 'rolling')) {
    machine.dataset.startedAt = String(performance.now());
    panel.dispatchEvent(new Event('reels:start'));
  }
  panel.classList.remove('result-reveal');
  panel.setAttribute('aria-busy', String(rolling));
  machine.dataset.phase = rolling ? 'rolling' : 'idle';
  machine.querySelector('h3').textContent = title;
  machine.querySelector('p').textContent = description;
}

function burst(machine) {
  const flare = document.createElement('div');
  flare.className = 'slot-burst';
  flare.setAttribute('aria-hidden', 'true');
  for (let index = 0; index < 18; index++) {
    const angle = index * Math.PI * 2 / 18;
    const distance = 78 + (index % 4) * 20;
    const spark = document.createElement('i');
    spark.style.setProperty('--spark-x', `${Math.cos(angle) * distance}px`);
    spark.style.setProperty('--spark-y', `${Math.sin(angle) * distance * 0.7}px`);
    spark.style.setProperty('--spark-delay', `${index % 3 * 32}ms`);
    flare.append(spark);
  }
  machine.querySelector('.slot-case').append(flare);
  machine.querySelector('h3').textContent = 'Особь пока неизвестна';
  machine.querySelector('p').textContent = '';
  machine.dataset.phase = 'triple';
}

export async function stopReels(panel, isCurrent, unknown = false, beforeReveal = () => {}) {
  const machine = panel.querySelector('.slot-machine');
  if (!machine || machine.dataset.phase !== 'rolling' || reducedMotion.matches || !Element.prototype.animate) {
    if (isCurrent()) {
      beforeReveal();
      panel.dispatchEvent(new Event('reels:stop'));
      if (unknown) panel.dispatchEvent(new Event('reels:triple'));
    }
    return;
  }
  const remaining = minSpinMs - (performance.now() - Number(machine.dataset.startedAt));
  if (remaining > 0) await pause(remaining);
  if (!isCurrent() || !machine.isConnected) return;
  beforeReveal();
  const tracks = [...machine.querySelectorAll('.slot-track')];
  const cycle = tracks[0]?.firstElementChild.getBoundingClientRect().height * symbols.length;
  if (!cycle) return;
  const positions = tracks.map(track => {
    const transform = getComputedStyle(track).transform;
    return transform === 'none' ? 0 : new DOMMatrixReadOnly(transform).m42;
  });
  machine.dataset.phase = 'settling';
  const animations = tracks.map((track, index) => {
    const start = positions[index];
    const rowHeight = cycle / symbols.length;
    // Each belt starts with a different symbol. Align all three horse marks.
    const matchAt = unknown ? (symbols.length - resting[index]) % symbols.length * rowHeight : 0;
    const end = (Math.floor((start + matchAt) / cycle) - 1) * cycle - matchAt;
    return track.animate([{ transform: `translateY(${start}px)` }, { transform: `translateY(${end}px)` }],
      { duration: 570 + index * 235, easing: 'cubic-bezier(.19,.72,.26,1)', fill: 'forwards' });
  });
  await Promise.allSettled(animations.map(animation => animation.finished));
  if (!isCurrent() || !machine.isConnected) return;
  panel.dispatchEvent(new Event('reels:stop'));
  if (unknown) {
    burst(machine);
    panel.dispatchEvent(new Event('reels:triple'));
    await pause(burstDurationMs);
  }
}

export function revealResult(panel, render, animate = true) {
  panel.dispatchEvent(new Event('reels:stop'));
  panel.classList.toggle('result-reveal', animate);
  render();
  panel.setAttribute('aria-busy', 'false');
}
