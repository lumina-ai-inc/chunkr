import { createRoot } from "react-dom/client";
import { StrictMode } from "react";
import { KcPage } from "./kc.gen";
import { getKcContextMock } from "./login/mocks/getKcContextMock";

const kcContext = getKcContextMock({
    pageId: "login-update-password.ftl",
    overrides: {}
});

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <KcPage kcContext={kcContext} />
    </StrictMode>
);
