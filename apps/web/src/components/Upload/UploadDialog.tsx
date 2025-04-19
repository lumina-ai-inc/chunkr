import { Dialog } from "@radix-ui/themes";
import { Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import { AuthContextProps } from "react-oidc-context";
import "./UploadDialog.css";
import UploadMain from "./UploadMain";
import { useState } from "react";

interface UploadDialogProps {
  auth: AuthContextProps;
  onUploadComplete?: () => void;
}

export default function UploadDialog({
  auth,
  onUploadComplete,
}: UploadDialogProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const isAuthenticated = auth.isAuthenticated;

  const handleOpenChange = (open: boolean) => {
    if (!isUploading) {
      setIsOpen(open);
    }
  };

  const handleUploadStart = () => {
    setIsUploading(true);
    setIsOpen(false);
  };

  const handleUploadComplete = () => {
    setIsUploading(false);
    onUploadComplete?.();
  };

  return (
    <Dialog.Root open={isOpen} onOpenChange={handleOpenChange}>
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
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M13.25 9.25H19.75L13.25 2.75V9.25Z"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M10 15.25L12.5 12.75L15 15.25"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M12.5 13.75V18.25"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
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
        <UploadMain
          isAuthenticated={isAuthenticated}
          onUploadSuccess={handleUploadComplete}
          onUploadStart={handleUploadStart}
        />
      </Dialog.Content>
    </Dialog.Root>
  );
}
