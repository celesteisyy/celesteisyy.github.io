#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo 'Run this installer with sudo.' >&2
  exit 1
fi
playground_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
install -d -o root -g root -m 0755 /var/www/cyth-playground /etc/nginx/snippets /etc/nginx/conf.d
for asset in index.html embed.html sign-in.html login-complete.html webui.css embed.js; do
  install -o root -g root -m 0644 "$playground_dir/$asset" "/var/www/cyth-playground/$asset"
done
install -o root -g root -m 0644 "$playground_dir/deploy/origin-guard.conf" /etc/nginx/conf.d/cyth-embed-origin.conf
install -o root -g root -m 0644 "$playground_dir/deploy/nginx.conf" /etc/nginx/snippets/cyth-playground.conf
echo 'Installed workspace files and the origin guard.'
echo 'Keep one include inside the existing cyth HTTPS server block:'
echo '    include /etc/nginx/snippets/cyth-playground.conf;'
echo 'Run: sudo nginx -t && sudo systemctl reload nginx'
echo 'Only then enable compose.embed.yaml with both existing Compose files.'
