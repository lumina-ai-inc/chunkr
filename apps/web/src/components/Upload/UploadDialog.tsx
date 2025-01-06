import { Dialog } from "@radix-ui/themes";
import { Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import UploadMain from "./UploadMain";
import { AuthContextProps } from "react-oidc-context";
import "./UploadDialog.css";

export default function UploadDialog({ auth }: { auth: AuthContextProps }) {
  const isAuthenticated = auth.isAuthenticated;

  return (
    <Dialog.Root>
      <Dialog.Trigger>
        <BetterButton>
          <svg
            width="18"
            height="18"
            viewBox="0 0 25 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <g clip-path="url(#clip0_113_1479)">
              <path
                d="M19.75 9.25V20.25C19.75 20.8 19.3 21.25 18.75 21.25H6.25C5.7 21.25 5.25 20.8 5.25 20.25V3.75C5.25 3.2 5.7 2.75 6.25 2.75H13.25"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M13.25 9.25H19.75L13.25 2.75V9.25Z"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M10 15.25L12.5 12.75L15 15.25"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M12.5 13.75V18.25"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </g>
            <defs>
              <clipPath id="clip0_113_1479">
                <rect
                  width="24"
                  height="24"
                  fill="white"
                  transform="translate(0.5)"
                />
              </clipPath>
            </defs>
          </svg>
          <Text size="2" weight="medium" style={{ color: "#FFF" }}>
            Create Task
          </Text>
        </BetterButton>
      </Dialog.Trigger>
      <Dialog.Content className="dialog-overlay">
        <Dialog.Close className="dialog-close">
          <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M11.7816 4.03157C12.0062 3.80702 12.0062 3.44295 11.7816 3.2184C11.5571 2.99385 11.193 2.99385 10.9685 3.2184L7.50005 6.68682L4.03164 3.2184C3.80708 2.99385 3.44301 2.99385 3.21846 3.2184C2.99391 3.44295 2.99391 3.80702 3.21846 4.03157L6.68688 7.49999L3.21846 10.9684C2.99391 11.193 2.99391 11.557 3.21846 11.7816C3.44301 12.0061 3.80708 12.0061 4.03164 11.7816L7.50005 8.31316L10.9685 11.7816C11.193 12.0061 11.5571 12.0061 11.7816 11.7816C12.0062 11.557 12.0062 11.193 11.7816 10.9684L8.31322 7.49999L11.7816 4.03157Z"
              fill="currentColor"
              fillRule="evenodd"
              clipRule="evenodd"
            ></path>
          </svg>
        </Dialog.Close>
        <UploadMain isAuthenticated={isAuthenticated} />
      </Dialog.Content>
    </Dialog.Root>
  );
}
