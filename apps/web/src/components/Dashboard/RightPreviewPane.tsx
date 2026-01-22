import { Flex, Text, Tabs, ScrollArea, Button, Table, Dialog, Checkbox, TextField } from "@radix-ui/themes";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "react-query";
import { getDeal, getDealFacts, getDealDocuments, DocumentResponse, deleteDeal, updateDealName } from "../../services/dealApi";
import { calculateUnderwriting, UnderwritingResult } from "../../services/underwritingApi";
import { getAllContacts, getContactsByType, getFamilyOfficeContacts, Contact, updateContact, deleteContact } from "../../services/contactApi";
import DealSummaryCard from "./DealSummaryCard";
import FactReviewDeal from "../FactReview/FactReviewDeal";
import UnderwritingDashboard from "../Underwriting/UnderwritingDashboard";
import InvestorPackage from "../InvestorPackage/InvestorPackage";
import OCRDocumentViewer from "../Documents/OCRDocumentViewer";
import { InterestTracker } from "../LiveShare/InterestTracker";
import { toast } from "react-hot-toast";
import "./RightPreviewPane.css";

interface RightPreviewPaneProps {
  dealId: string | null;
  previewType: "empty" | "document" | "analysis" | "memo" | "facts" | "underwriting";
  selectedContactType?: string | null;
  selectedLiveShareId?: string | null;
  onDealDeleted?: () => void;
}

export default function RightPreviewPane({
  dealId,
  previewType,
  selectedContactType,
  selectedLiveShareId,
  onDealDeleted,
}: RightPreviewPaneProps) {
  
  // Early return if live share is selected
  if (selectedLiveShareId) {
    return <InterestTracker shareId={selectedLiveShareId} />;
  }
  const [viewingDocument, setViewingDocument] = useState<DocumentResponse | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [selectedContactIds, setSelectedContactIds] = useState<Set<string>>(new Set());
  const [editingContactId, setEditingContactId] = useState<string | null>(null);
  const [editForm, setEditForm] = useState<Partial<Contact>>({});
  const queryClient = useQueryClient();

  const { data: deal } = useQuery(
    ["deal", dealId],
    () => (dealId ? getDeal(dealId) : null),
    { enabled: !!dealId }
  );

  const { data: _facts } = useQuery(
    ["facts", dealId],
    () => (dealId ? getDealFacts(dealId) : null),
    { enabled: !!dealId }
  );

  const { data: documents } = useQuery(
    ["documents", dealId],
    () => (dealId ? getDealDocuments(dealId) : null),
    { enabled: !!dealId }
  );

  // Check if deal is ready for underwriting
  const isReadyForUnderwriting = deal?.status === "ready_for_underwriting" || deal?.status === "underwriting_complete";

  const { 
    data: underwriting, 
    isLoading: isLoadingUnderwriting,
    isError: isUnderwritingError,
    refetch: refetchUnderwriting 
  } = useQuery<UnderwritingResult | null>(
    ["underwriting", dealId],
    () => (dealId ? calculateUnderwriting(dealId) : null),
    { 
      enabled: !!dealId && (previewType === "underwriting" || isReadyForUnderwriting),
      // Retry on error to handle cases where underwriting hasn't been calculated yet
      retry: 1,
      // Refetch when tab changes to underwriting
      refetchOnWindowFocus: previewType === "underwriting",
    }
  );

  // Get contacts if contact type is selected
  const contacts = selectedContactType
    ? selectedContactType === "all"
      ? getAllContacts()
      : selectedContactType === "investors"
      ? getContactsByType("investor")
      : selectedContactType === "institutional"
      ? getContactsByType("institutional")
      : selectedContactType === "family_office"
      ? getFamilyOfficeContacts()
      : []
    : [];

  // Delete deal mutation
  const deleteDealMutation = useMutation(
    (dealIdToDelete: string) => deleteDeal(dealIdToDelete),
    {
      onSuccess: () => {
        queryClient.invalidateQueries("deals");
        toast.success("Deal deleted successfully");
        if (onDealDeleted) {
          onDealDeleted();
        }
      },
      onError: (error: any) => {
        console.error("Error deleting deal:", error);
        toast.error("Failed to delete deal");
      },
    }
  );

  // Update deal name mutation
  const updateDealNameMutation = useMutation(
    (dealName: string) => (dealId ? updateDealName(dealId, dealName) : Promise.reject("No deal ID")),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(["deal", dealId]);
        queryClient.invalidateQueries("deals");
        toast.success("Deal name updated successfully");
      },
      onError: (error: any) => {
        console.error("Error updating deal name:", error);
        toast.error("Failed to update deal name");
      },
    }
  );

  const handleDeleteDeal = () => {
    if (dealId) {
      deleteDealMutation.mutate(dealId);
      setShowDeleteDialog(false);
    }
  };

  // Show contacts view if contact type is selected (and no deal is selected)
  if (selectedContactType && !dealId) {
    const handleSelectContact = (contactId: string) => {
      const newSet = new Set(selectedContactIds);
      if (newSet.has(contactId)) {
        newSet.delete(contactId);
      } else {
        newSet.add(contactId);
      }
      setSelectedContactIds(newSet);
    };

    const handleSelectAll = () => {
      if (selectedContactIds.size === contacts.length) {
        setSelectedContactIds(new Set());
      } else {
        setSelectedContactIds(new Set(contacts.map((c) => c.contact_id)));
      }
    };

    const handleDeleteSelected = () => {
      if (selectedContactIds.size === 0) {
        toast.error("Please select contacts to delete");
        return;
      }

      let deletedCount = 0;
      selectedContactIds.forEach((contactId) => {
        if (deleteContact(contactId)) {
          deletedCount++;
        }
      });

      if (deletedCount > 0) {
        toast.success(`Deleted ${deletedCount} contact(s)`);
        setSelectedContactIds(new Set());
        queryClient.invalidateQueries("allContacts");
        queryClient.invalidateQueries("accreditedInvestors");
        queryClient.invalidateQueries("institutionalContacts");
        queryClient.invalidateQueries("familyOfficeContacts");
        // Force re-render by reloading contacts
        window.location.reload();
      }
    };

    const handleEditContact = (contact: Contact) => {
      setEditingContactId(contact.contact_id);
      setEditForm({
        first_name: contact.first_name,
        last_name: contact.last_name,
        email: contact.email,
        title: contact.title,
        company: contact.company,
        mobile_phone: contact.mobile_phone,
      });
    };

    const handleSaveEdit = (contactId: string) => {
      if (updateContact(contactId, editForm)) {
        toast.success("Contact updated successfully");
        setEditingContactId(null);
        setEditForm({});
        queryClient.invalidateQueries("allContacts");
        queryClient.invalidateQueries("accreditedInvestors");
        queryClient.invalidateQueries("institutionalContacts");
        queryClient.invalidateQueries("familyOfficeContacts");
        window.location.reload();
      } else {
        toast.error("Failed to update contact");
      }
    };

    const handleCancelEdit = () => {
      setEditingContactId(null);
      setEditForm({});
    };

    return (
      <Flex
        direction="column"
        style={{
          width: "650px",
          height: "100vh",
          backgroundColor: "#f8f9fa",
          flexShrink: 0,
        }}
      >
        <Flex
          direction="column"
          p="24px"
          style={{ height: "100%", display: "flex", flexDirection: "column" }}
        >
          <Text size="5" weight="bold" mb="16px" style={{ flexShrink: 0 }}>
            {selectedContactType === "all"
              ? "All Contacts"
              : selectedContactType === "investors"
              ? "Accredited Investors"
              : selectedContactType === "institutional"
              ? "Institutional"
              : "Family Offices"}
          </Text>

          {contacts.length > 0 ? (
            <>
              <ScrollArea style={{ flex: 1, minHeight: 0 }} scrollbars="vertical">
                <Table.Root>
                  <Table.Header>
                    <Table.Row>
                      <Table.ColumnHeaderCell style={{ width: "40px" }}>
                        <Checkbox
                          checked={selectedContactIds.size === contacts.length && contacts.length > 0}
                          onCheckedChange={handleSelectAll}
                        />
                      </Table.ColumnHeaderCell>
                      <Table.ColumnHeaderCell>First Name</Table.ColumnHeaderCell>
                      <Table.ColumnHeaderCell>Last Name</Table.ColumnHeaderCell>
                      <Table.ColumnHeaderCell>Email</Table.ColumnHeaderCell>
                      <Table.ColumnHeaderCell style={{ width: "100px" }}>Actions</Table.ColumnHeaderCell>
                    </Table.Row>
                  </Table.Header>
                  <Table.Body>
                    {contacts.map((contact: Contact) => (
                      <Table.Row key={contact.contact_id}>
                        <Table.Cell>
                          <Checkbox
                            checked={selectedContactIds.has(contact.contact_id)}
                            onCheckedChange={() => handleSelectContact(contact.contact_id)}
                          />
                        </Table.Cell>
                        <Table.Cell>
                          {editingContactId === contact.contact_id ? (
                            <TextField.Root
                              size="1"
                              value={editForm.first_name || ""}
                              onChange={(e) => setEditForm({ ...editForm, first_name: e.target.value })}
                              placeholder="First Name"
                            />
                          ) : (
                            <Text size="2" weight="medium">
                              {contact.first_name || "-"}
                            </Text>
                          )}
                        </Table.Cell>
                        <Table.Cell>
                          {editingContactId === contact.contact_id ? (
                            <TextField.Root
                              size="1"
                              value={editForm.last_name || ""}
                              onChange={(e) => setEditForm({ ...editForm, last_name: e.target.value })}
                              placeholder="Last Name"
                            />
                          ) : (
                            <Text size="2" weight="medium">
                              {contact.last_name || "-"}
                            </Text>
                          )}
                        </Table.Cell>
                        <Table.Cell>
                          {editingContactId === contact.contact_id ? (
                            <TextField.Root
                              size="1"
                              value={editForm.email || ""}
                              onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
                              placeholder="Email"
                            />
                          ) : (
                            <Text size="2" style={{ color: "#666" }}>
                              {contact.email || "-"}
                            </Text>
                          )}
                        </Table.Cell>
                        <Table.Cell>
                          {editingContactId === contact.contact_id ? (
                            <Flex gap="2">
                              <Button
                                size="1"
                                onClick={() => handleSaveEdit(contact.contact_id)}
                              >
                                Save
                              </Button>
                              <Button
                                size="1"
                                variant="soft"
                                onClick={handleCancelEdit}
                              >
                                Cancel
                              </Button>
                            </Flex>
                          ) : (
                            <Button
                              size="1"
                              variant="soft"
                              onClick={() => handleEditContact(contact)}
                            >
                              Edit
                            </Button>
                          )}
                        </Table.Cell>
                      </Table.Row>
                    ))}
                  </Table.Body>
                </Table.Root>
              </ScrollArea>
              <Flex gap="3" justify="end" mt="16px" style={{ flexShrink: 0, paddingTop: "16px", borderTop: "1px solid #e0e0e0" }}>
                <Button
                  size="2"
                  variant="soft"
                  onClick={handleSelectAll}
                >
                  {selectedContactIds.size === contacts.length ? "Deselect All" : "Select All"}
                </Button>
                <Button
                  size="2"
                  variant="soft"
                  color="red"
                  onClick={handleDeleteSelected}
                  disabled={selectedContactIds.size === 0}
                >
                  Delete Selected ({selectedContactIds.size})
                </Button>
              </Flex>
            </>
          ) : (
            <Flex
              direction="column"
              align="center"
              justify="center"
              p="40px"
              style={{ flex: 1 }}
            >
              <Text size="3" style={{ color: "#999" }}>
                No contacts found
              </Text>
              <Text size="2" style={{ color: "#999", marginTop: "8px" }}>
                Import contacts to get started
              </Text>
            </Flex>
          )}
        </Flex>
      </Flex>
    );
  }

  // Empty state - show when no deal is selected and no contact type is selected
  if (!dealId && !selectedContactType) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        style={{
          width: "400px",
          height: "100vh",
          padding: "40px",
          textAlign: "center",
          backgroundColor: "#f8f9fa",
          flexShrink: 0,
        }}
      >
        <Text size="5" weight="medium" style={{ marginBottom: "12px" }}>
          Quick Start Guide
        </Text>
        <Text size="3" style={{ color: "#666", lineHeight: "1.6" }}>
          Select a deal from the left panel to view its details, or create a new deal to get started.
        </Text>
        <Flex direction="column" gap="12px" mt="24px" style={{ width: "100%" }}>
          <Text size="2" weight="medium" style={{ textAlign: "left" }}>
            What you can do:
          </Text>
          <Text size="2" style={{ textAlign: "left", color: "#666" }}>
            • Upload property documents
          </Text>
          <Text size="2" style={{ textAlign: "left", color: "#666" }}>
            • Extract facts automatically
          </Text>
          <Text size="2" style={{ textAlign: "left", color: "#666" }}>
            • Run underwriting analysis
          </Text>
          <Text size="2" style={{ textAlign: "left", color: "#666" }}>
            • Generate investor memos
          </Text>
        </Flex>
      </Flex>
    );
  }

  // Content with tabs
  return (
    <Flex
      direction="column"
      style={{
        width: "800px", // Increased by 20% (650 * 1.2 = 780)
        height: "100vh",
        backgroundColor: "#f8f9fa",
        flexShrink: 0,
      }}
    >
      <Tabs.Root defaultValue="memo" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
        <Tabs.List style={{ 
          padding: "10px", 
          backgroundColor: "#fff", 
          borderBottom: "0px solid #e0e0e0",
          display: "flex",
          width: "100%"
        }}>
          <Tabs.Trigger value="memo" style={{ flex: 1 }}>Memo</Tabs.Trigger>
          <Tabs.Trigger value="underwriting" style={{ flex: 1 }}>Analysis</Tabs.Trigger>
          <Tabs.Trigger value="facts" style={{ flex: 1 }}>Facts</Tabs.Trigger>
          <Tabs.Trigger value="documents" style={{ flex: 1 }}>Documents</Tabs.Trigger>
          <Tabs.Trigger value="analysis" style={{ flex: 1 }}>Summary</Tabs.Trigger>
        </Tabs.List>

        <ScrollArea style={{ flex: 1, minHeight: 0 }} scrollbars="vertical">
          {/* Analysis Tab */}
          <Tabs.Content value="analysis" style={{ padding: "16px" }}>
            {deal && (
              <Flex direction="column" gap="16px">
                <DealSummaryCard
                  deal={deal}
                  onDealNameUpdate={(newName) => {
                    updateDealNameMutation.mutate(newName);
                  }}
                  metrics={
                    underwriting
                      ? {
                          grossRent: 100000,
                          noi: underwriting.noi,
                          dscr: underwriting.dscr,
                          capRate: underwriting.cap_rate,
                        }
                      : undefined
                  }
                />
                <Flex justify="end" mt="16px">
                  <Button
                    size="2"
                    variant="soft"
                    color="red"
                    onClick={() => setShowDeleteDialog(true)}
                    disabled={deleteDealMutation.isLoading}
                  >
                    Delete Deal
                  </Button>
                </Flex>
              </Flex>
            )}
          </Tabs.Content>

          {/* Documents Tab */}
          <Tabs.Content value="documents" style={{ padding: "32px" }}>
            <Flex direction="column" gap="12px">
              <Text size="6" weight="medium">
                Parsed Documents
              </Text>
              {documents && documents.length > 0 ? (
                documents.map((doc) => {
                  // #region agent log: Check each document
                  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'RightPreviewPane.tsx:465',message:'Rendering document',data:{fileName:doc.file_name,status:doc.status,hasStorageLocation:!!doc.storage_location,storageLocationPrefix:doc.storage_location?.substring(0,100),willShowViewSource:(doc.status==="completed"||doc.status==="processed")},timestamp:Date.now(),sessionId:'debug-session',runId:'view-source-debug',hypothesisId:'H23'})}).catch(()=>{});
                  // #endregion
                  return (
                  <Flex
                    key={doc.document_id}
                    justify="between"
                    align="center"
                    p="12px"
                    style={{
                      backgroundColor: "#fff",
                      borderRadius: "8px",
                      border: "1px solid #e0e0e0",
                    }}
                  >
                    <Flex direction="column" gap="1" style={{ flex: 1 }}>
                      <Text size="2" weight="medium">
                        {doc.file_name}
                      </Text>
                      <Text size="1" style={{ color: "#666" }}>
                        {doc.status} • {doc.page_count || 0} pages
                      </Text>
                    </Flex>
                    {doc.status === "completed" || doc.status === "processed" || doc.status === "succeeded" ? (
                      <Flex
                        align="center"
                        gap="4px"
                        onClick={() => setViewingDocument(doc)}
                        style={{ 
                          cursor: "pointer",
                          color: "#666",
                          fontSize: "13px"
                        }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                          <polyline points="15 3 21 3 21 9"></polyline>
                          <line x1="10" y1="14" x2="21" y2="3"></line>
                        </svg>
                        <Text size="3" style={{ color: "#666" }}>View Source</Text>
                      </Flex>
                    ) : (
                      <Text size="2" style={{ color: "#999", fontSize: "12px" }}>
                        {doc.status === "processing" || doc.status === "pending" 
                          ? "⏳ Processing..." 
                          : doc.status === "failed" 
                          ? "❌ Failed" 
                          : "Not available"}
                      </Text>
                    )}
                  </Flex>
                );
                })
              ) : (
                <Text size="2" style={{ color: "#999" }}>
                  No documents uploaded yet
                </Text>
              )}
            </Flex>
          </Tabs.Content>

          {/* Facts Tab */}
          <Tabs.Content value="facts" style={{ padding: "16px" }}>
            {dealId && <FactReviewDeal dealId={dealId} />}
          </Tabs.Content>

          {/* Underwriting Tab */}
          <Tabs.Content value="underwriting" style={{ padding: "16px" }}>
            {dealId && isLoadingUnderwriting && (
              <Flex direction="column" align="center" justify="center" p="24px" gap="12px">
                <Text size="3" style={{ color: "#666" }}>
                  Calculating underwriting metrics...
                </Text>
              </Flex>
            )}
            {dealId && isUnderwritingError && (
              <Flex direction="column" align="center" justify="center" p="24px" gap="12px">
                <Text size="3" style={{ color: "#e74c3c" }}>
                  Failed to calculate underwriting. Please try again.
                </Text>
                <Button size="2" onClick={() => refetchUnderwriting()}>
                  Retry
                </Button>
              </Flex>
            )}
            {dealId && underwriting && !isLoadingUnderwriting && (
              <UnderwritingDashboard dealId={dealId} />
            )}
            {dealId && !underwriting && !isLoadingUnderwriting && !isUnderwritingError && (
              <Flex direction="column" align="center" justify="center" p="24px" gap="12px">
                <Text size="3" style={{ color: "#999", textAlign: "center" }}>
                  Run underwriting analysis in Facts into see metrics
                </Text>
                <Text size="2" style={{ color: "#999", textAlign: "center" }}>
                  Go to the Facts tab and click "Run Analysis →"
                </Text>
              </Flex>
            )}
          </Tabs.Content>

          {/* Memo Tab */}
          <Tabs.Content value="memo" style={{ padding: "16px" }}>
            {dealId && <InvestorPackage />}
          </Tabs.Content>
        </ScrollArea>
      </Tabs.Root>

      {/* OCR Document Viewer Dialog */}
      {dealId && (
        <OCRDocumentViewer
          document={viewingDocument}
          dealId={dealId}
          open={!!viewingDocument}
          onOpenChange={(open) => !open && setViewingDocument(null)}
        />
      )}

      {/* Delete Deal Confirmation Dialog */}
      <Dialog.Root open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <Dialog.Content style={{ maxWidth: "400px" }}>
          <Dialog.Title>Delete Deal</Dialog.Title>
          <Dialog.Description size="2" mb="4">
            Are you sure you want to delete "{deal?.deal_name}"? This action cannot be undone.
          </Dialog.Description>
          <Flex gap="3" mt="4" justify="end">
            <Dialog.Close>
              <Button variant="soft" color="gray">
                Cancel
              </Button>
            </Dialog.Close>
            <Button
              variant="solid"
              color="red"
              onClick={handleDeleteDeal}
              disabled={deleteDealMutation.isLoading}
            >
              {deleteDealMutation.isLoading ? "Deleting..." : "Delete"}
            </Button>
          </Flex>
        </Dialog.Content>
      </Dialog.Root>
    </Flex>
  );
}

