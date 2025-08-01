# Development stage
FROM node:18 as development
WORKDIR /app
COPY apps/web/package.json apps/web/pnpm-lock.yaml ./
RUN npm install -g pnpm
RUN pnpm install --frozen-lockfile
COPY apps/web/ .
EXPOSE 5173
CMD ["pnpm", "dev", "--host", "0.0.0.0"]

# Build stage
FROM node:18 as build

WORKDIR /app

COPY apps/web/ .

RUN git init && \
    git config user.email "build@docker.com" && \
    git config user.name "Docker Build" && \
    git add . && \
    git commit -m "Initial commit for keycloakify"

# Install pnpm, turbo, and required type definitions
RUN npm install -g pnpm turbo
RUN rm -rf node_modules && pnpm install --frozen-lockfile

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

# Process index.html for environment variables
cp /usr/share/nginx/html/index.html /usr/share/nginx/html/index.html.template
envsubst < /usr/share/nginx/html/index.html.template > /usr/share/nginx/html/index.html 
rm /usr/share/nginx/html/index.html.template

# Process nginx.conf for environment variables
cp /etc/nginx/conf.d/default.conf /etc/nginx/conf.d/default.conf.template
envsubst '$CSP_CONNECT_SRC_ADDITIONAL $CSP_IMG_SRC_ADDITIONAL' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf
rm /etc/nginx/conf.d/default.conf.template

EOF

RUN chmod +x /docker-entrypoint.d/01-insert-window-variable.sh

EXPOSE 8000
