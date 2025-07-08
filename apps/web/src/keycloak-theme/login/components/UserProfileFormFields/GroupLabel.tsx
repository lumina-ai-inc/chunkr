/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/UserProfileFormFields/GroupLabel.tsx" --revert
 */

import { assert } from "tsafe/assert";
import type { Attribute } from "@keycloakify/login-ui/KcContext";
import { useI18n } from "../../i18n";
import { Text } from "@radix-ui/themes";

export function GroupLabel(props: {
    attribute: Attribute;
    groupNameRef: {
        current: string;
    };
}) {
    const { attribute, groupNameRef } = props;

    const { advancedMsg } = useI18n();

    if (attribute.group?.name !== groupNameRef.current) {
        groupNameRef.current = attribute.group?.name ?? "";

        if (groupNameRef.current !== "") {
            assert(attribute.group !== undefined);

            return (
                <div
                    className="space-y-4 mb-6"
                    {...Object.fromEntries(
                        Object.entries(attribute.group.html5DataAnnotations).map(([key, value]) => [
                            `data-${key}`,
                            value
                        ])
                    )}
                >
                    {(() => {
                        const groupDisplayHeader = attribute.group.displayHeader ?? "";
                        const groupHeaderText =
                            groupDisplayHeader !== ""
                                ? advancedMsg(groupDisplayHeader)
                                : attribute.group.name;

                        return (
                            <div>
                                <h3
                                    id={`header-${attribute.group.name}`}
                                    className="text-lg font-bold text-gray-900 mb-2"
                                >
                                    {groupHeaderText}
                                </h3>
                            </div>
                        );
                    })()}
                    {(() => {
                        const groupDisplayDescription = attribute.group.displayDescription ?? "";

                        if (groupDisplayDescription !== "") {
                            const groupDescriptionText = advancedMsg(groupDisplayDescription);

                            return (
                                <div>
                                    <Text
                                        as="p"
                                        size="2"
                                        color="gray"
                                        id={`description-${attribute.group.name}`}
                                        className="text-gray-600"
                                    >
                                        {groupDescriptionText}
                                    </Text>
                                </div>
                            );
                        }

                        return null;
                    })()}
                </div>
            );
        }
    }

    return null;
}
