import { Flex, Text, Button, Dialog, TextField, RadioGroup } from "@radix-ui/themes";
import { useState, useEffect } from "react";
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
  const [showNewDealDialog, setShowNewDealDialog] = useState(false);
  const [newDealName, setNewDealName] = useState("");
  const [newDealType, setNewDealType] = useState<'rental_income' | 'value_add'>('rental_income');

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

  const handleNewDealClick = () => {
    // #region agent log
    void fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DashboardThreePane.tsx:61',message:'handleNewDealClick called',data:{currentShowNewDealDialog:showNewDealDialog},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
    // #endregion
    setShowNewDealDialog(true);
  };

  const handleCreateDeal = async () => {
    // #region agent log
    void fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DashboardThreePane.tsx:65',message:'handleCreateDeal called',data:{newDealName,newDealType,showNewDealDialog},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
    // #endregion
    if (!newDealName.trim()) return;
    await createNewDeal(newDealName, newDealType);
    setShowNewDealDialog(false);
    setNewDealName("");
    setNewDealType('rental_income');
  };

  // Log when dialog opens/closes
  useEffect(() => {
    if (showNewDealDialog) {
      // #region agent log
      void fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DashboardThreePane.tsx:useEffect',message:'Dialog opened',data:{showNewDealDialog,newDealType,newDealName},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
      // #endregion
    }
  }, [showNewDealDialog, newDealType, newDealName]);

  return (
    <Flex style={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Left Pane: 240px fixed */}
      <LeftNavPane
        selectedDealId={currentDealId}
        onSelectDeal={handleSelectDeal}
        onNewDeal={handleNewDealClick}
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

      {/* New Deal Dialog */}
      <Dialog.Root open={showNewDealDialog} onOpenChange={(open) => {
        // #region agent log
        void fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DashboardThreePane.tsx:104',message:'Dialog onOpenChange',data:{open,showNewDealDialog,newDealType,newDealName},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
        // #endregion
        setShowNewDealDialog(open);
        if (!open) {
          setNewDealName("");
          setNewDealType('rental_income');
        }
      }}>
        <Dialog.Content style={{ maxWidth: 450 }}>
          <Dialog.Title>Create New Deal</Dialog.Title>
          <Dialog.Description size="2" mb="4">
            Enter a name for your new underwriting deal.
          </Dialog.Description>

          <Flex direction="column" gap="3">
            <Flex direction="column" gap="2">
              <Text as="div" size="2" weight="bold">
                Deal Strategy
              </Text>
              <RadioGroup.Root 
                value={newDealType} 
                onValueChange={(val) => setNewDealType(val as 'rental_income' | 'value_add')}
              >
                <Flex direction="column" gap="2">
                  <Text as="label" size="2" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <RadioGroup.Item value="rental_income" />
                    Rental Income (Buy & Hold)
                  </Text>
                  <Text as="label" size="2" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <RadioGroup.Item value="value_add" />
                    Value-Add (Rehab)
                  </Text>
                </Flex>
              </RadioGroup.Root>
            </Flex>
            
            <Flex direction="column" gap="2">
              <Text as="div" size="2" weight="bold">
                Deal Name
              </Text>
              <TextField.Root
                value={newDealName}
                onChange={(e) => setNewDealName(e.target.value)}
                placeholder="e.g., 123 Main St Portfolio"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newDealName.trim()) {
                    handleCreateDeal();
                  }
                }}
              />
            </Flex>
          </Flex>

          <Flex gap="3" mt="4" justify="end">
            <Dialog.Close>
              <Button variant="soft" color="gray">
                Cancel
              </Button>
            </Dialog.Close>
            <Button
              onClick={handleCreateDeal}
              disabled={!newDealName.trim()}
            >
              Create Deal
            </Button>
          </Flex>
        </Dialog.Content>
      </Dialog.Root>
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

