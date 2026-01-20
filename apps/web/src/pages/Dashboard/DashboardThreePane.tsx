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
  const [selectedLiveShareId, setSelectedLiveShareId] = useState<string | null>(null);

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

  const handleSelectDeal = (dealId: string) => {
    selectDeal(dealId);
    setSelectedContactType(null); // Clear contact selection when deal is selected
    setSelectedLiveShareId(null); // Clear live share selection when deal is selected
  };

  const handleSelectContactType = (type: string | null) => {
    setSelectedContactType(type);
    // Note: Cannot clear deal selection since selectDeal requires non-null string
    // Deal will remain selected when viewing contacts
  };

  const handleSelectLiveShare = (shareId: string) => {
    setSelectedLiveShareId(shareId);
    // Note: We don't clear deal selection via selectDeal since it requires a non-null string
    // The UI will show InterestTracker instead of deal content when selectedLiveShareId is set
    setSelectedContactType(null); // Clear contact selection
  };

  return (
    <Flex style={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Left Pane: 240px fixed */}
      <LeftNavPane
        selectedDealId={currentDealId}
        onSelectDeal={handleSelectDeal}
        onNewDeal={createNewDeal}
        onSelectLiveShare={handleSelectLiveShare}
        selectedLiveShareId={selectedLiveShareId}
      />

      {/* Middle Pane: Flexible width (min 400px) */}
      <MiddleChatPane
        dealId={currentDealId}
        messages={currentMessages}
        onSendMessage={handleSendMessage}
        onFileUpload={handleFileUpload}
      />

      {/* Right Pane: 400px fixed */}
      <RightPreviewPane 
        dealId={currentDealId} 
        previewType={previewType} 
        selectedContactType={selectedContactType}
        selectedLiveShareId={selectedLiveShareId}
        onDealDeleted={() => {
          // Deal deleted - nothing to do, ChatContext handles it
        }}
      />
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

