import { useEffect, type ReactNode } from "react";
import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import { useSetClassName } from "@keycloakify/login-ui/tools/useSetClassName";
import { useInitializeTemplate } from "./useInitializeTemplate";
import { useI18n } from "../../i18n";
import { useKcContext } from "../../KcContext";
import { Flex, Text, Heading } from "@radix-ui/themes";
import footerText from "../../../../assets/footer/footer-text-comp.png";
import { Header } from "@radix-ui/themes/components/table";

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
  }, [documentTitle, kcContext.realm.displayName, msgStr]);

  useSetClassName({ qualifiedName: "body", className: bodyClassName });

  const { isReadyToRender } = useInitializeTemplate();
  if (!isReadyToRender) return null;

  return (
    <div className="flex flex-col min-h-screen w-full">
      <header className="px-6 py-4">
        <img src={"https://t7nw0vdho0.ufs.sh/f/wvRR96mLyWoQvKBudtA6DA9Zfpq0VcuyBgYFEJ8olnmWwMSP"} alt="Chunkr Logo" className="h-8 object-contain" />
      </header>
      <div className="flex-1 flex items-center justify-center pt-10">
        <div className="text-white rounded-lg p-4 sm:p-6 md:p-8 lg:p-10 w-full max-w-sm sm:max-w-md md:max-w-lg transition-all duration-300 ease-in-out">        
          <Header className="flex flex-row items-center justify-start gap-2 mb-6">
            {kcContext.auth?.showUsername &&
            !kcContext.auth?.showResetCredentials ? (
              <div className="flex flex-col gap-3">
                <Text as="label">{kcContext.auth.attemptedUsername}</Text>
                <a
                  href={kcContext.url.loginRestartFlowUrl}
                  aria-label={msgStr("restartLoginTooltip")}
                  className="text-sm"
                >
                  <Text>
                    <i /> {msg("restartLoginTooltip")}
                  </Text>
                </a>
              </div>
            ) : (
              <Heading size="5" id="kc-page-title" weight="bold">
                {headerNode}
              </Heading>
            )}

            {displayRequiredFields && (
              <div className="flex flex-col mt-4">
                <Text size="2" color="gray">
                  <span className="text-red-500">*</span>{" "}
                  {msg("requiredFields")}
                </Text>
              </div>
            )}
          </Header>
          <main>
            <div className="flex flex-col gap-4 mt-4">
              {displayMessage &&
                kcContext.message &&
                (kcContext.message.type !== "warning" ||
                  !kcContext.isAppInitiatedAction) && (
                  <div
                    className={`border-l-4 pl-3 py-2 ${
                      kcContext.message.type === "error"
                        ? "border-red-500 text-red-600"
                        : kcContext.message.type === "warning"
                        ? "border-orange-500 text-orange-600"
                        : "border-green-500 text-green-600"
                    }`}
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
                    className="text-blue-600 hover:text-blue-800 underline"
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
                <div className="mt-4">
                  <Text size="2" color="gray">
                    {infoNode}
                  </Text>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>
      <Flex direction="row" align="center" justify="center" className="footer-logo w-full">
        <img 
          src={footerText} 
          alt="chunkr" 
          className="h-32 sm:h-40 md:h-48 lg:h-56 xl:h-64 max-w-full object-contain transition-all duration-300 ease-in-out" 
        />
      </Flex>
    </div>
  );
}