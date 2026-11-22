// Дополнительный модуль «полевой книги»: родословная и дневник взросления.
// Работает параллельно с казино-интерфейсом app.js и не трогает его логику.
import { createPedigreeDialog } from './pedigree-ui.js';
import './growth.js';

const state = { horses: [], canEditCatalog: false };

async function request(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Не удалось выполнить запрос.');
  return data;
}

function updateCatalog() {
  document.dispatchEvent(new CustomEvent('catalog-updated', { detail: state.horses }));
}

async function load() {
  try {
    const [config, catalog] = await Promise.all([request('/api/config'), request('/api/horses')]);
    state.canEditCatalog = config.canEditCatalog;
    state.horses = catalog.horses;
    window.openPedigree = createPedigreeDialog(state, request, updateCatalog);
    document.dispatchEvent(new CustomEvent('catalog-updated', { detail: state.horses }));
  } catch (error) {
    console.error('Fieldbook unavailable:', error.message);
  }
}

document.addEventListener('catalog-updated', event => {
  if (Array.isArray(event.detail)) state.horses = event.detail;
});

load();
