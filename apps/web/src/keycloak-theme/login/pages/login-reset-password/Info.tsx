import { assert } from "tsafe/assert";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";

export function Info() {
    const { kcContext } = useKcContext();
    assert(kcContext.pageId === "login-reset-password.ftl");
    const { msg } = useI18n();

    return (
        <div className="text-sm text-gray-300">
            {kcContext.realm.duplicateEmailsAllowed
                ? msg("emailInstructionUsername")
                : msg("emailInstruction")}
        </div>
    )
}