import Keycloak, { KeycloakConfig, KeycloakInitOptions } from 'keycloak-js';

const keycloakConfig: KeycloakConfig = {
    url: import.meta.env.VITE_KEYCLOAK_URL,
    realm: import.meta.env.VITE_KEYCLOAK_REALM,
    clientId: import.meta.env.VITE_KEYCLOAK_CLIENT_ID,
};

const initOptions: KeycloakInitOptions = {
    onLoad: 'check-sso',
    // redirectUri: import.meta.env.VITE_KEYCLOAK_REDIRECT_URI,
    checkLoginIframe: true,
    pkceMethod: 'S256'
}

export const keycloak = new Keycloak(keycloakConfig);

export async function initializeKeycloak(): Promise<boolean> {
    return keycloak.init(initOptions)
        .then((auth) => {
            if (!auth) {
                console.error("Authentication failed");
                return false;
            } else {
                console.log("Keycloak authenticated:", keycloak.authenticated);
                console.log("Keycloak id token parsed:", keycloak.idTokenParsed);
                scheduleTokenRefresh();
                return true;
            }
        })
        .catch((error) => {
            console.error('Failed to initialize Keycloak:', error);
            return false;
        });
}

function scheduleTokenRefresh() {
    setTimeout(() => {
        refreshToken();
    }, 60000);
}

function refreshToken() {
    keycloak.updateToken(70).then((refreshed) => {
        if (refreshed) {
            console.debug('Token refreshed');
        } else {
            if (!keycloak.tokenParsed || !keycloak.timeSkew || !keycloak.tokenParsed.exp) {
                console.warn('Token information is incomplete');
                return;
            }
            const expiresIn = Math.round(keycloak.tokenParsed.exp + keycloak.timeSkew - new Date().getTime() / 1000);
            console.warn(`Token not refreshed, valid for ${expiresIn} seconds`);
        }
        scheduleTokenRefresh();
    }).catch((error) => {
        console.error('Failed to refresh token:', error);
        // Attempt to re-authenticate
        keycloak.login();
    });
}