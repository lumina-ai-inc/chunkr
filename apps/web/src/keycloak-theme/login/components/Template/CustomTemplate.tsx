import { useEffect, type ReactNode } from "react";
import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import { useSetClassName } from "@keycloakify/login-ui/tools/useSetClassName";
import { useInitializeTemplate } from "./useInitializeTemplate";
import { useI18n } from "../../i18n";
import { useKcContext } from "../../KcContext";
import { Card, Flex, Text, Heading } from "@radix-ui/themes";

export function Template(props: {
  displayInfo?: boolean;
  displayMessage?: boolean;
  displayRequiredFields?: boolean;
  headerNode: ReactNode;
  socialProvidersNode?: ReactNode;
  infoNode?: ReactNode;
  documentTitle?: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  const {
    displayInfo = false,
    displayMessage = true,
    displayRequiredFields = false,
    headerNode,
    socialProvidersNode = null,
    infoNode = null,
    documentTitle,
    bodyClassName,
    children,
  } = props;

  const { kcContext } = useKcContext();
  const { msg, msgStr } = useI18n();

  useEffect(() => {
    document.title =
      documentTitle ?? msgStr("loginTitle", kcContext.realm.displayName);
  }, []);

  useSetClassName({ qualifiedName: "body", className: bodyClassName });

  const { isReadyToRender } = useInitializeTemplate();
  if (!isReadyToRender) return null;

  return (
    <Flex
      height="100vh"
      width="100%"
      align="center"
      justify="center"
      direction="column"
    >
      <Card size="4" style={{ width: 416, padding: 24 }}>
        <header>
          {kcContext.auth?.showUsername &&
          !kcContext.auth?.showResetCredentials ? (
            <Flex direction="column" gap="3">
              <Text as="label">{kcContext.auth.attemptedUsername}</Text>
              <a
                href={kcContext.url.loginRestartFlowUrl}
                aria-label={msgStr("restartLoginTooltip")}
                style={{ fontSize: "0.875rem" }}
              >
                <Text>
                  <i /> {msg("restartLoginTooltip")}
                </Text>
              </a>
            </Flex>
          ) : (
            <Heading as="h1" size="4" id="kc-page-title">
              {headerNode}
            </Heading>
          )}

          {displayRequiredFields && (
            <Flex direction="column" mt="4">
              <Text size="2" color="gray">
                <span style={{ color: "red" }}>*</span>{" "}
                {msg("requiredFields")}
              </Text>
            </Flex>
          )}
        </header>

        <main>
          <Flex direction="column" gap="4" mt="4">
            {displayMessage &&
              kcContext.message &&
              (kcContext.message.type !== "warning" ||
                !kcContext.isAppInitiatedAction) && (
                <div
                  style={{
                    borderLeft: "4px solid",
                    paddingLeft: "0.75rem",
                    color:
                      kcContext.message.type === "error"
                        ? "crimson"
                        : kcContext.message.type === "warning"
                        ? "orange"
                        : "green",
                  }}
                >
                  <Text
                    dangerouslySetInnerHTML={{
                      __html: kcSanitize(kcContext.message.summary),
                    }}
                  />
                </div>
              )}

            {children}

            {kcContext.auth?.showTryAnotherWayLink && (
              <form
                id="kc-select-try-another-way-form"
                action={kcContext.url.loginAction}
                method="post"
              >
                <input type="hidden" name="tryAnotherWay" value="on" />
                <a
                  href="#"
                  onClick={() => {
                    document.forms[
                      "kc-select-try-another-way-form" as never
                    ].submit();
                    return false;
                  }}
                >
                  {msg("doTryAnotherWay")}
                </a>
              </form>
            )}

            {socialProvidersNode}

            {displayInfo && (
              <div style={{ marginTop: "1rem" }}>
                <Text size="2" color="gray">
                  {infoNode}
                </Text>
              </div>
            )}
          </Flex>
        </main>
      </Card>
    </Flex>
  );
}