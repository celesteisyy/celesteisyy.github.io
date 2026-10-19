#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo 'Run this installer with sudo.' >&2
  exit 1
fi
playground_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
install -d -o root -g root -m 0755 /var/www/cyth-playground /etc/nginx/snippets
for asset in index.html webui.css webui.js; do
  install -o root -g root -m 0644 "$playground_dir/$asset" "/var/www/cyth-playground/$asset"
done
install -o root -g root -m 0644 "$playground_dir/deploy/nginx.conf" /etc/nginx/snippets/cyth-playground.conf
echo 'Installed workspace files. Add this line inside the existing cyth HTTPS server block:'
echo '    include /etc/nginx/snippets/cyth-playground.conf;'
echo 'Then run: sudo nginx -t && sudo systemctl reload nginx'
