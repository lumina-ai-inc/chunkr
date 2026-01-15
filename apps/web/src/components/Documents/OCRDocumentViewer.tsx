import { Flex, Text, Dialog, Button, Card, Badge, Spinner } from "@radix-ui/themes";
import { DocumentResponse, FactResponse } from "../../services/dealApi";
import { getDealFacts, pollDocumentStatus } from "../../services/dealApi";
import { useQuery } from "react-query";
import { isPreexistingMockDeal } from "../../services/mockDealData";
import { useEffect, useState } from "react";

interface OCRDocumentViewerProps {
  document: DocumentResponse | null;
  dealId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function OCRDocumentViewer({
  document,
  dealId,
  open,
  onOpenChange,
}: OCRDocumentViewerProps) {
  const [pollingDocument, setPollingDocument] = useState<DocumentResponse | null>(null);
  const [processingStatus, setProcessingStatus] = useState<string>("");

  const { data: facts } = useQuery<FactResponse[]>(
    ["deal-facts", dealId],
    () => getDealFacts(dealId),
    { enabled: !!dealId && open }
  );

  // Poll document status if it's processing
  useEffect(() => {
    if (!document || !open) return;
    
    // Poll even for mock deals since they may have real tasks
    if (document.status === "processing" || document.status === "pending") {
      setProcessingStatus("Processing document...");
      
      pollDocumentStatus(
        dealId,
        document.document_id,
        (status, _ocrOutput) => {
          setProcessingStatus(status === "completed" ? "Processing complete!" : "Processing...");
        }
      )
        .then((updatedDoc: DocumentResponse) => {
          setPollingDocument(updatedDoc);
          setProcessingStatus(updatedDoc.status === "completed" ? "Processing complete!" : "Processing failed");
        })
        .catch((error: any) => {
          console.error("Polling error:", error);
          setProcessingStatus("Processing timeout - please refresh");
        });
    } else {
      setPollingDocument(document);
    }
  }, [document, dealId, open]);

  if (!document) return null;
  
  const displayDocument = pollingDocument || document;

  const documentFacts = facts?.filter(
    (f) => f.source_citation.document === document.file_name
  ) || [];

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content style={{ maxWidth: "90vw", maxHeight: "90vh" }}>
        <Dialog.Title>{document.file_name || "Source Document"}</Dialog.Title>
        <Dialog.Description size="2" mb="4">
          OCR Preview - {document.page_count || 0} pages
        </Dialog.Description>

        <Flex direction="column" gap="3" style={{ maxHeight: "70vh", overflow: "auto" }}>
          {displayDocument.status === "processing" || displayDocument.status === "pending" ? (
            // Show processing status
            <Flex direction="column" gap="3" align="center" justify="center" style={{ minHeight: "200px" }}>
              <Spinner size="3" />
              <Text size="3" weight="medium">
                {processingStatus || "⏳ Processing Document"}
              </Text>
              <Text size="2" color="gray">
                The document is being processed. OCR preview will be available once complete.
              </Text>
            </Flex>
          ) : displayDocument.status === "failed" ? (
            // Show error state
            <Card style={{ background: "#ffe0e0", borderColor: "#ff4444", padding: "16px" }}>
              <Flex direction="column" gap="2">
                <Text size="3" weight="bold" style={{ color: "#cc0000" }}>
                  ❌ Processing Failed
                </Text>
                <Text size="2" style={{ color: "#cc0000" }}>
                  There was an error processing this document. Please try uploading again.
                </Text>
              </Flex>
            </Card>
          ) : isPreexistingMockDeal(dealId) ? (
            // Pre-existing demo deals - show sample document with disclaimer
            <Flex direction="column" gap="3">
              <Card style={{ background: "#fff3cd", borderColor: "#ffc107", padding: "16px" }}>
                <Flex direction="column" gap="2">
                  <Text size="3" weight="bold" style={{ color: "#856404" }}>
                    ⚠️ Mock Data Preview
                  </Text>
                  <Text size="2" style={{ color: "#856404" }}>
                    This is a sample document preview for demonstration purposes. 
                    In production, this would display the actual uploaded document with OCR text overlay.
                  </Text>
                </Flex>
              </Card>

              <Card style={{ padding: "24px", background: "#f8f9fa", border: "1px solid #e0e0e0" }}>
                <Flex direction="column" gap="3">
                  <Flex direction="column" gap="1">
                    <Text size="4" weight="bold">
                      {document.file_name || "Sample Document"}
                    </Text>
                    <Text size="2" color="gray">
                      Document Type: {document.document_type || "N/A"}
                    </Text>
                    {document.page_count && (
                      <Text size="2" color="gray">
                        Pages: {document.page_count}
                      </Text>
                    )}
                    <Text size="2" color="gray">
                      Status: {document.status || "N/A"}
                    </Text>
                  </Flex>

                  {documentFacts.length > 0 && (
                    <Flex direction="column" gap="2" mt="3">
                      <Text size="3" weight="medium">
                        Extracted Facts from this Document:
                      </Text>
                      {documentFacts.map((fact) => (
                        <Card key={fact.fact_id} style={{ padding: "12px", background: "#fff" }}>
                          <Flex direction="column" gap="1">
                            <Flex justify="between" align="center">
                              <Text size="2" weight="medium">{fact.label}</Text>
                              <Badge
                                color={
                                  fact.status === "approved"
                                    ? "green"
                                    : fact.status === "missing"
                                    ? "red"
                                    : "yellow"
                                }
                              >
                                {fact.status === "approved"
                                  ? "Verified"
                                  : fact.status === "missing"
                                  ? "Missing"
                                  : "Needs Review"}
                              </Badge>
                            </Flex>
                            {fact.value && (
                              <Text size="3" weight="bold">
                                {fact.value} {fact.unit || ""}
                              </Text>
                            )}
                            <Text size="1" color="gray">
                              Page {fact.source_citation.page}
                              {fact.source_citation.line && `, Line: ${fact.source_citation.line}`}
                            </Text>
                          </Flex>
                        </Card>
                      ))}
                    </Flex>
                  )}

                  <Flex direction="column" gap="1" mt="3" p="3" style={{ background: "#e9ecef", borderRadius: "4px" }}>
                    <Text size="2" weight="medium">Document Metadata:</Text>
                    <Text size="1" color="gray">
                      Created:{" "}
                      {document.created_at
                        ? new Date(document.created_at).toLocaleDateString()
                        : "N/A"}
                    </Text>
                    {document.extracted_at && (
                      <Text size="1" color="gray">
                        Extracted: {new Date(document.extracted_at).toLocaleDateString()}
                      </Text>
                    )}
                  </Flex>
                </Flex>
              </Card>
            </Flex>
          ) : document.storage_location ? (
            // Real data - show actual document
            <iframe
              src={document.storage_location}
              style={{
                width: "100%",
                height: "600px",
                border: "1px solid #e0e0e0",
                borderRadius: "4px",
              }}
              title={document.file_name}
            />
          ) : (
            <Flex
              direction="column"
              align="center"
              justify="center"
              p="8"
              gap="2"
              style={{ minHeight: "400px" }}
            >
              <Text size="4" color="gray">
                📄 {document.file_name}
              </Text>
              <Text size="2" color="gray">
                Document preview not available. The document will be available once uploaded and processed.
              </Text>
            </Flex>
          )}
        </Flex>

        <Flex gap="3" mt="4" justify="end">
          <Dialog.Close>
            <Button variant="soft">Close</Button>
          </Dialog.Close>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}

