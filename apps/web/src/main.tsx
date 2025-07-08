if (window.kcContext !== undefined) {
  import("./keycloak-theme/main");
} else if (import.meta.env.VITE_KC_DEV === "true") {
  import("./keycloak-theme/main.dev");
} else {
  import("./main-app");
}
