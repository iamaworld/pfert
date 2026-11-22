#!/bin/zsh
set -e

cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
trap 'echo; read -r "?Нажмите Enter, чтобы закрыть окно..."' EXIT

if ! command -v node >/dev/null; then
  echo "Не найден Node.js. Установите Node.js 20 или новее."
  exit 1
fi
if ! command -v cloudflared >/dev/null; then
  echo "Не найден cloudflared. Установите Cloudflare Tunnel."
  exit 1
fi

echo "Запускаю защищённый общий доступ к Тахи..."
npm run share
echo "Общий доступ остановлен."
