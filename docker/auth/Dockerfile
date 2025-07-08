FROM node:18 as keycloakify_jar_builder

# Install required dependencies including Maven and JDK
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get install -y maven git;

# Copy package files first
COPY apps/web/package.json apps/web/pnpm-lock.yaml /opt/app/
WORKDIR /opt/app

# Install pnpm
RUN npm install -g pnpm

# Copy source code before installing dependencies
COPY apps/web/ /opt/app/

# Initialize git repository to satisfy keycloakify requirements
RUN git init && \
    git config user.email "build@docker.com" && \
    git config user.name "Docker Build" && \
    git add . && \
    git commit -m "Initial commit for keycloakify"

# Install dependencies (now that source code is available and git is initialized)
RUN pnpm install --frozen-lockfile

# Build the Keycloak theme
RUN IS_KC_BUILD=true pnpm run build && pnpm exec keycloakify build

# Builder stage - prepare Keycloak with theme
FROM quay.io/keycloak/keycloak:25.0.2 AS builder

WORKDIR /opt/keycloak

# Copy the theme JAR to providers directory
COPY --from=keycloakify_jar_builder /opt/app/dist_keycloak/keycloak-theme-for-kc-all-other-versions.jar /opt/keycloak/providers/

# Build the optimized Keycloak distribution
RUN /opt/keycloak/bin/kc.sh build

# Final stage - runtime image
FROM quay.io/keycloak/keycloak:25.0.2

# Copy the built Keycloak from builder stage
COPY --from=builder /opt/keycloak/ /opt/keycloak/

# Set the entrypoint
ENTRYPOINT ["/opt/keycloak/bin/kc.sh", "start-dev"]
