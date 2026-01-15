import { Flex } from "@radix-ui/themes";
import { useState } from "react";
import { ChatProvider, useChatContext } from "../../contexts/ChatContext";
import LeftNavPane from "../../components/Dashboard/LeftNavPane";
import MiddleChatPane from "../../components/Dashboard/MiddleChatPane";
import RightPreviewPane from "../../components/Dashboard/RightPreviewPane";
import "./DashboardThreePane.css";

function DashboardContent() {
  const {
    currentDealId,
    chatSessions,
    selectDeal,
    createNewDeal,
    sendMessage,
    uploadFile,
    previewType,
  } = useChatContext();
  
  const [selectedContactType, setSelectedContactType] = useState<string | null>(null);

  const currentMessages = currentDealId
    ? chatSessions.get(currentDealId) || []
    : [];

  const handleSendMessage = (message: string) => {
    if (currentDealId) {
      sendMessage(currentDealId, message);
    }
  };

  const handleFileUpload = (file: File) => {
    if (currentDealId) {
      uploadFile(currentDealId, file);
    }
  };

  return (
    <Flex style={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Left Pane: 240px fixed */}
      <LeftNavPane
        selectedDealId={currentDealId}
        onSelectDeal={selectDeal}
        onNewDeal={createNewDeal}
        selectedContactType={selectedContactType}
        onSelectContactType={setSelectedContactType}
      />

      {/* Middle Pane: Flexible width (min 400px) */}
      <MiddleChatPane
        dealId={currentDealId}
        messages={currentMessages}
        onSendMessage={handleSendMessage}
        onFileUpload={handleFileUpload}
      />

      {/* Right Pane: 400px fixed */}
      <RightPreviewPane dealId={currentDealId} previewType={previewType} />
    </Flex>
  );
}

export default function DashboardThreePane() {
  return (
    <ChatProvider>
      <DashboardContent />
    </ChatProvider>
  );
}

