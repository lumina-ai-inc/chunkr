import { useEffect, type ReactNode } from "react";
import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
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
    children,
  } = props;

  const { kcContext } = useKcContext();
  const { msg, msgStr } = useI18n();

  useEffect(() => {
    document.title =
      documentTitle ?? msgStr("loginTitle", kcContext.realm.displayName);
  }, [documentTitle, kcContext.realm.displayName, msgStr]);

  const { isReadyToRender } = useInitializeTemplate();
  if (!isReadyToRender) return null;

  return (
    <div className="flex flex-col min-h-screen w-full">
      <Flex direction="row" justify="center" className="header-container">
        <div className="header">
          <header className="w-full h-fit z-1 px-5 py-3 flex items-center">
            <div className="header-logo-container">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="30"
                height="30"
                viewBox="0 0 30 30"
                fill="none"
              >
                <path
                  d="M7.35 12.225C8.03148 12.4978 8.77803 12.5646 9.4971 12.4171C10.2162 12.2695 10.8761 11.9142 11.3952 11.3952C11.9142 10.8761 12.2695 10.2162 12.4171 9.4971C12.5646 8.77803 12.4978 8.03148 12.225 7.35C13.0179 7.13652 13.7188 6.6687 14.2201 6.01836C14.7214 5.36802 14.9954 4.57111 15 3.75C17.225 3.75 19.4001 4.4098 21.2502 5.64597C23.1002 6.88213 24.5422 8.63914 25.3936 10.6948C26.2451 12.7505 26.4679 15.0125 26.0338 17.1948C25.5998 19.3771 24.5283 21.3816 22.955 22.955C21.3816 24.5283 19.3771 25.5998 17.1948 26.0338C15.0125 26.4679 12.7505 26.2451 10.6948 25.3936C8.63914 24.5422 6.88213 23.1002 5.64597 21.2502C4.4098 19.4001 3.75 17.225 3.75 15C4.57111 14.9954 5.36802 14.7214 6.01836 14.2201C6.6687 13.7188 7.13652 13.0179 7.35 12.225Z"
                  stroke="url(#paint0_linear_236_740)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <defs>
                  <linearGradient
                    id="paint0_linear_236_740"
                    x1="15"
                    y1="3.75"
                    x2="15"
                    y2="26.25"
                    gradientUnits="userSpaceOnUse"
                  >
                    <stop stopColor="white" />
                    <stop offset="1" stopColor="#DCE4DD" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
          </header>
        </div>
      </Flex>
      <div className="flex-1 flex items-center justify-center pt-10">
        <div className="text-white rounded-lg p-4 sm:p-6 md:p-8 lg:p-10 w-full max-w-sm sm:max-w-md md:max-w-lg transition-all duration-300 ease-in-out">        
          <Header className="flex flex-row items-center justify-between gap-2 mb-6">
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
                <Text size="2" className="text-gray-300">
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
                  className={`border-l-4 pl-3 py-2 rounded-md text-sm ${
                    kcContext.message.type === "error"
                      ? "border-red-500 text-red-600 bg-red-500/10"
                      : kcContext.message.type === "warning"
                      ? "border-orange-500 text-orange-600 bg-orange-500/10"
                      : kcContext.message.type === "success"
                      ? "border-green-500 text-green-600 bg-green-500/10"
                      : "border-blue-500 text-blue-600 bg-blue-500/10"
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
                    {infoNode}
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