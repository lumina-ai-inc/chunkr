# Build stage
FROM node:18 as build

# # Handle environment variables
# COPY apps/web/.env .env
# ENV $(cat .env | xargs)
# RUN rm .env

WORKDIR /app

COPY apps/web/ .

# Install pnpm and dependencies
RUN npm install -g pnpm turbo
RUN pnpm install

# Build the project using Turbo
RUN pnpm run build

# Production stage
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY apps/web/robots.txt /usr/share/nginx/html/robots.txt

COPY <<'EOF' /docker-entrypoint.d/01-insert-window-variable.sh
#!/bin/sh

set -eu

cp /usr/share/nginx/html/index.html /usr/share/nginx/html/index.html.template
envsubst < /usr/share/nginx/html/index.html.template > /usr/share/nginx/html/index.html 
rm /usr/share/nginx/html/index.html.template

EOF

RUN chmod +x /docker-entrypoint.d/01-insert-window-variable.sh

EXPOSE 8000

CMD ["nginx", "-g", "daemon off;"]