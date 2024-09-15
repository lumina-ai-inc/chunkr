#!/bin/sh

# Replace placeholders with environment variables
sed -i 's~__API_URL__~'"$API_URL"'~g' /app/dist/env-config.js
sed -i 's~__KEYCLOAK_URL__~'"$KEYCLOAK_URL"'~g' /app/dist/env-config.js
sed -i 's~__KEYCLOAK_REALM__~'"$KEYCLOAK_REALM"'~g' /app/dist/env-config.js
sed -i 's~__KEYCLOAK_CLIENT_ID__~'"$KEYCLOAK_CLIENT_ID"'~g' /app/dist/env-config.js
sed -i 's~__KEYCLOAK_REDIRECT_URI__~'"$KEYCLOAK_REDIRECT_URI"'~g' /app/dist/env-config.js
sed -i 's~__KEYCLOAK_POST_LOGOUT_REDIRECT_URI__~'"$KEYCLOAK_POST_LOGOUT_REDIRECT_URI"'~g' /app/dist/env-config.js
sed -i 's~__STRIPE_API_KEY__~'"$STRIPE_API_KEY"'~g' /app/dist/env-config.js

exec "$@"