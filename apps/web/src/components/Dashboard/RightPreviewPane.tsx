import { Flex, Text, Tabs, ScrollArea, Button } from "@radix-ui/themes";
import { useState } from "react";
import { useQuery } from "react-query";
import { getDeal, getDealFacts, getDealDocuments, DocumentResponse } from "../../services/dealApi";
import { calculateUnderwriting, UnderwritingResult } from "../../services/underwritingApi";
import DealSummaryCard from "./DealSummaryCard";
import FactReviewDeal from "../FactReview/FactReviewDeal";
import UnderwritingDashboard from "../Underwriting/UnderwritingDashboard";
import InvestorPackage from "../InvestorPackage/InvestorPackage";
import OCRDocumentViewer from "../Documents/OCRDocumentViewer";
import "./RightPreviewPane.css";

interface RightPreviewPaneProps {
  dealId: string | null;
  previewType: "empty" | "document" | "analysis" | "memo" | "facts" | "underwriting";
}

export default function RightPreviewPane({
  dealId,
  previewType,
}: RightPreviewPaneProps) {
  const [viewingDocument, setViewingDocument] = useState<DocumentResponse | null>(null);

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

  // Empty state
  if (!dealId || previewType === "empty") {
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
        width: "650px",
        height: "100vh",
        backgroundColor: "#f8f9fa",
        flexShrink: 0,
      }}
    >
      <Tabs.Root defaultValue="memo" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
        <Tabs.List style={{ padding: "16px 16px 0", backgroundColor: "#fff", borderBottom: "1px solid #e0e0e0" }}>
          <Tabs.Trigger value="memo">Memo</Tabs.Trigger>
          <Tabs.Trigger value="underwriting">Analysis</Tabs.Trigger>
          <Tabs.Trigger value="facts">Facts</Tabs.Trigger>
          <Tabs.Trigger value="documents">Documents</Tabs.Trigger>
          <Tabs.Trigger value="analysis">Summary</Tabs.Trigger>
        </Tabs.List>

        <ScrollArea style={{ flex: 1 }} scrollbars="vertical">
          {/* Analysis Tab */}
          <Tabs.Content value="analysis" style={{ padding: "16px" }}>
            {deal && (
              <DealSummaryCard
                deal={deal}
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
            )}
          </Tabs.Content>

          {/* Documents Tab */}
          <Tabs.Content value="documents" style={{ padding: "16px" }}>
            <Flex direction="column" gap="12px">
              <Text size="4" weight="medium">
                Documents
              </Text>
              {documents && documents.length > 0 ? (
                documents.map((doc) => (
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
                    <Button
                      size="2"
                      variant="outline"
                      onClick={() => setViewingDocument(doc)}
                      style={{ flexShrink: 0 }}
                    >
                      View Source
                    </Button>
                  </Flex>
                ))
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
                  Run underwriting analysis to see metrics
                </Text>
                <Text size="2" style={{ color: "#999", textAlign: "center" }}>
                  Go to the Facts tab and click "Run Underwriting →"
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
    </Flex>
  );
}

