const preferenceKey = 'takhi-reels-muted';

function createFirecracker(context) {
  const rate = context.sampleRate;
  const buffer = context.createBuffer(1, Math.round(0.52 * rate), rate);
  const samples = buffer.getChannelData(0);
  const cracks = [0.049, 0.086, 0.121, 0.173, 0.219, 0.274, 0.343];
  let previousNoise = 0;
  for (let index = 0; index < samples.length; index++) {
    const time = index / rate;
    const noise = Math.random() * 2 - 1;
    const bright = (noise - previousNoise * 0.75) * 0.55;
    previousNoise = noise;
    const pop = Math.sin(2 * Math.PI * (168 * time - 240 * time * time)) * Math.exp(-44 * time) * 0.45;
    let envelope = Math.exp(-36 * time) * 0.55;
    for (const [i, crack] of cracks.entries()) {
      const elapsed = time - crack;
      if (elapsed >= 0 && elapsed < 0.038) envelope += Math.exp(-135 * elapsed) * (0.24 - i * 0.023);
    }
    samples[index] = (pop + bright * envelope) * Math.min(1, time * 600);
  }
  return buffer;
}

function combineFirecrackers(context, shot, beatMs) {
  const interval = Math.round(context.sampleRate * beatMs / 1000);
  const buffer = context.createBuffer(1, shot.length + 2 * interval, context.sampleRate);
  const samples = buffer.getChannelData(0);
  const pop = shot.getChannelData(0);
  for (let beat = 0; beat < 3; beat++) {
    for (let i = 0; i < pop.length; i++) samples[i + beat * interval] += pop[i] * (0.95 - beat * 0.12);
  }
  return buffer;
}

export function connectReelAudio(panel, toggle) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    toggle.hidden = true;
    return () => {};
  }

  let enabled = true;
  try { enabled = localStorage.getItem(preferenceKey) !== '1'; } catch { /* Storage can be blocked. */ }
  let context;
  let recording;
  let playback;
  let firecracker;
  let victory;
  let effect;
  let rotating = false;
  let version = 0;

  function updateToggle() {
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.textContent = enabled ? 'Звук вкл.' : 'Звук выкл.';
  }

  function prepare() {
    if (!enabled) return;
    try {
      context ??= new AudioContextClass();
      if (context.state !== 'running') context.resume().catch(() => {});
    } catch {
      // The investigation works even if this browser cannot play sound.
    }
  }

  function stop() {
    version++;
    delete toggle.dataset.playing;
    if (!playback) return;
    const { player, gain } = playback;
    const now = context.currentTime;
    gain.gain.cancelScheduledValues(now);
    gain.gain.setValueAtTime(gain.gain.value, now);
    gain.gain.linearRampToValueAtTime(0, now + 0.24);
    player.stop(now + 0.25);
    playback = null;
  }

  async function start() {
    if (!enabled || !context || document.hidden) return;
    const currentVersion = ++version;
    try {
      recording ??= fetch('/reel-sound.mp3').then(response => {
        if (!response.ok) throw new Error('Audio unavailable');
        return response.arrayBuffer();
      }).then(bytes => context.decodeAudioData(bytes)).catch(error => {
        recording = null;
        throw error;
      });
      const buffer = await recording;
      if (context.state !== 'running') await context.resume();
      if (currentVersion !== version || !rotating || !enabled || document.hidden) return;
      const player = context.createBufferSource();
      const gain = context.createGain();
      player.buffer = buffer;
      player.loop = true;
      player.connect(gain).connect(context.destination);
      const now = context.currentTime;
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(0.45, now + 0.12);
      player.start(now);
      playback = { player, gain };
      toggle.dataset.playing = 'true';
      player.onended = () => { player.disconnect(); gain.disconnect(); };
    } catch {
      delete toggle.dataset.playing;
    }
  }

  function stopEffect() {
    if (!effect) return;
    const current = effect;
    effect = null;
    delete toggle.dataset.effect;
    current.stop();
  }

  function playEffect(kind, beatMs) {
    if (!enabled || document.hidden || context?.state !== 'running') return;
    stopEffect();
    firecracker ??= createFirecracker(context);
    const source = context.createBufferSource();
    const gain = context.createGain();
    if (kind === 'win') victory ??= combineFirecrackers(context, firecracker, beatMs);
    source.buffer = kind === 'win' ? victory : firecracker;
    gain.gain.value = 0.6;
    source.connect(gain).connect(context.destination);
    effect = source;
    toggle.dataset.effect = kind;
    source.onended = () => {
      source.disconnect();
      gain.disconnect();
      if (effect === source) {
        effect = null;
        delete toggle.dataset.effect;
      }
    };
    source.start();
  }

  panel.addEventListener('reels:start', () => {
    rotating = true;
    stopEffect();
    stop();
    start();
  });
  panel.addEventListener('reels:stop', () => {
    rotating = false;
    stop();
  });
  panel.addEventListener('reels:triple', () => playEffect('triple'));
  panel.addEventListener('prediction:win', event => playEffect('win', event.detail.beatMs));
  toggle.addEventListener('click', () => {
    enabled = !enabled;
    updateToggle();
    try { localStorage.setItem(preferenceKey, enabled ? '0' : '1'); } catch { /* Session preference is still used. */ }
    if (!enabled) { stop(); stopEffect(); }
    else {
      prepare();
      if (rotating) start();
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { stop(); stopEffect(); }
    else if (rotating) start();
  });
  updateToggle();
  return prepare;
}
