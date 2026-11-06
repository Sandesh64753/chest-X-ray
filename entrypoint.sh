#!/bin/bash
set -e

# Default to 10000 (standard port for Render Web Services)
export PORT="${PORT:-10000}"

echo "=========================================================="
echo " Starting ChestVision AI Single-Container Service"
echo " Nginx Public Port : ${PORT}"
echo " Uvicorn Internal  : 127.0.0.1:8000"
echo "=========================================================="

# Substitute ${PORT} in Nginx template
sed "s/\${PORT}/${PORT}/g" /etc/nginx/templates/frontend.nginx.conf.template > /etc/nginx/conf.d/default.conf

# Remove default Debian site configuration if present
rm -f /etc/nginx/sites-enabled/default

# Verify Nginx configuration syntax
nginx -t

# Ensure log and pid directories exist
mkdir -p /var/log/supervisor /var/run

# Hand over to Supervisord as PID 1
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
