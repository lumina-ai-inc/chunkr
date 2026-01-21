import { useState } from "react";
import { Flex, Text, Card, Button, TextField, Badge } from "@radix-ui/themes";
import { useQuery, useMutation, useQueryClient } from "react-query";
import {
  getDealFacts,
  createFact,
  updateFact,
  approveFacts,
  resetFacts,
  updateDealStatus,
  FactResponse,
} from "../../services/dealApi";
import toast from "react-hot-toast";
import "./FactReviewDeal.css";

interface FactReviewDealProps {
  dealId: string;
  onFactsApproved?: () => void;
}

type FactStatus = "missing" | "needs_review" | "verified";

const FactReviewDeal = ({ dealId, onFactsApproved }: FactReviewDealProps) => {
  const [editedFacts, setEditedFacts] = useState<
    Record<string, { value: string; unit?: string }>
  >({});
  const [selectedFacts, setSelectedFacts] = useState<Set<string>>(new Set());
  const [showAddFactForm, setShowAddFactForm] = useState(false);
  const [newFactLabel, setNewFactLabel] = useState("");
  const [newFactValue, setNewFactValue] = useState("");
  const [newFactUnit, setNewFactUnit] = useState("");
  const queryClient = useQueryClient();

  // Initial seeded fields
  const SEEDED_FIELDS = [
    { label: "Gross Rent", unit: "$", type: "currency" },
    { label: "Operating Expenses", unit: "$", type: "currency" },
    { label: "Loan Amount", unit: "$", type: "currency" },
    { label: "Interest Rate", unit: "%", type: "percentage" },
    { label: "Loan Term", unit: "years", type: "number" },
  ];


  const {
    data: facts,
    isLoading,
    isError,
    refetch,
  } = useQuery<FactResponse[]>({
    queryKey: ["deal-facts", dealId],
    queryFn: () => getDealFacts(dealId),
  });

  // Initialize factsList early to avoid temporal dead zone errors
  const factsList = facts || [];
  const hasLockedFacts = factsList.some((f) => f.locked);

  const createFactMutation = useMutation({
    mutationFn: (data: {
      label: string;
      value: string;
      unit?: string;
    }) => createFact(dealId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deal-facts", dealId] });
      queryClient.invalidateQueries({ queryKey: ["deal", dealId] });
      setNewFactLabel("");
      setNewFactValue("");
      setNewFactUnit("");
      setShowAddFactForm(false);
      toast.success("Fact added");
    },
    onError: (error: any) => {
      toast.error(error.message || "Failed to create fact");
    },
  });

  const updateMutation = useMutation({
    mutationFn: (data: {
      factId: string;
      value: string;
      unit?: string;
    }) => updateFact(dealId, data.factId, { value: data.value }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deal-facts", dealId] });
      toast.success("Fact updated");
    },
    onError: (error: any) => {
      toast.error(error.message || "Failed to update fact");
    },
  });

  const approveMutation = useMutation({
    mutationFn: (factIds: string[]) => approveFacts(dealId, factIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deal-facts", dealId] });
      queryClient.invalidateQueries({ queryKey: ["deal", dealId] });
      toast.success("Facts verified and locked");
      setSelectedFacts(new Set());
      // Don't automatically trigger onFactsApproved here
    },
    onError: (error: any) => {
      toast.error(error.message || "Failed to approve facts");
    },
  });

  const resetMutation = useMutation({
    mutationFn: () => resetFacts(dealId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deal-facts", dealId] });
      queryClient.invalidateQueries({ queryKey: ["deal", dealId] });
      toast.success("Facts reset to editable state");
      setEditedFacts({});
      setSelectedFacts(new Set());
    },
    onError: (error: any) => {
      toast.error(error.message || "Failed to reset facts");
    },
  });

  const handleValueChange = (factId: string, value: string) => {
    setEditedFacts({
      ...editedFacts,
      [factId]: { value, unit: editedFacts[factId]?.unit },
    });
  };

  const handleUnitChange = (factId: string, unit: string) => {
    setEditedFacts({
      ...editedFacts,
      [factId]: { value: editedFacts[factId]?.value || "", unit },
    });
  };

  const saveFact = (factId: string) => {
    const edited = editedFacts[factId];
    if (edited) {
      updateMutation.mutate({
        factId,
        value: edited.value,
        unit: edited.unit,
      });
      const newEdited = { ...editedFacts };
      delete newEdited[factId];
      setEditedFacts(newEdited);
    }
  };

  const toggleFactSelection = (factId: string) => {
    const newSelected = new Set(selectedFacts);
    if (newSelected.has(factId)) {
      newSelected.delete(factId);
    } else {
      newSelected.add(factId);
    }
    setSelectedFacts(newSelected);
  };

  const handleApproveSelected = () => {
    if (selectedFacts.size === 0) {
      toast.error("Please select facts to verify");
      return;
    }
    approveMutation.mutate(Array.from(selectedFacts));
  };

  const handleVerifyAll = () => {
    if (!facts || facts.length === 0) {
      toast.error("No facts to verify");
      return;
    }
    const unlocked = facts.filter((f) => !f.locked);
    if (unlocked.length === 0) {
      toast.success("All facts are already verified");
      return;
    }
    // Select all and verify in one action
    const allIds = unlocked.map((f) => f.fact_id);
    setSelectedFacts(new Set(allIds));
    approveMutation.mutate(allIds);
  };

  const handleAddFact = () => {
    if (!newFactLabel.trim() || !newFactValue.trim()) {
      toast.error("Please provide both field name and value");
      return;
    }

    // Validate Loan Term (should be 20-30 years)
    if (newFactLabel.trim().toLowerCase() === "loan term") {
      const termValue = parseFloat(newFactValue.trim());
      if (isNaN(termValue) || termValue < 20 || termValue > 30) {
        toast.error("Loan Term must be between 20 and 30 years");
        return;
      }
    }

    // Validate Interest Rate (should be a floating point number)
    if (newFactLabel.trim().toLowerCase() === "interest rate") {
      const rateValue = parseFloat(newFactValue.trim());
      if (isNaN(rateValue) || rateValue < 0 || rateValue > 100) {
        toast.error("Interest Rate must be a valid number between 0 and 100");
        return;
      }
    }

    createFactMutation.mutate({
      label: newFactLabel.trim(),
      value: newFactValue.trim(),
      unit: newFactUnit.trim() || undefined,
    });
  };

  const handleAddSeededField = (field: { label: string; unit: string; type: string }) => {
    setNewFactLabel(field.label);
    setNewFactUnit(field.unit);
    setNewFactValue("");
    setShowAddFactForm(true);
  };

  // Check which seeded fields are missing
  const missingSeededFields = SEEDED_FIELDS.filter(
    (field) => !factsList.some((f) => f.label.toLowerCase() === field.label.toLowerCase())
  );

  const handleRunUnderwriting = async () => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'FactReviewDeal.tsx:handleRunUnderwriting',message:'handleRunUnderwriting called',data:{dealId,factsCount:facts?.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
    // #endregion
    if (!facts || facts.length === 0) {
      toast.error("No facts available for underwriting");
      return;
    }
    
    // Update deal status to ready_for_underwriting
    try {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'FactReviewDeal.tsx:handleRunUnderwriting',message:'Calling updateDealStatus',data:{dealId,status:'ready_for_underwriting'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      await updateDealStatus(dealId, "ready_for_underwriting");
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'FactReviewDeal.tsx:handleRunUnderwriting',message:'updateDealStatus completed, invalidating queries',data:{dealId},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      queryClient.invalidateQueries({ queryKey: ["deals"] });
      queryClient.invalidateQueries({ queryKey: ["deal", dealId] });
      // Invalidate underwriting query to trigger refetch
      queryClient.invalidateQueries({ queryKey: ["underwriting", dealId] });
    } catch (error) {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'FactReviewDeal.tsx:handleRunUnderwriting',message:'updateDealStatus failed',data:{dealId,error:error instanceof Error?error.message:String(error)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      console.error("Failed to update deal status:", error);
    }
    
    // Run underwriting even with unverified facts
    toast.success("Running underwriting analysis...");
    if (onFactsApproved) onFactsApproved();
  };

  // Determine fact status based on new rules
  const getFactStatus = (fact: FactResponse): FactStatus => {
    // If locked/approved, it's verified
    if (fact.locked || fact.status === "approved") {
      return "verified";
    }
    
    // If no confidence score or very low, it's missing
    if (!fact.confidence_score || fact.confidence_score < 0.5) {
      return "missing";
    }
    
    // If single source or moderate confidence, needs review
    if (fact.confidence_score < 0.9) {
      return "needs_review";
    }
    
    // High confidence with multiple sources = verified
    return "verified";
  };

  const getStatusDisplay = (status: FactStatus) => {
    switch (status) {
      case "missing":
        return { icon: "🔴", label: "Missing", color: "red" };
      case "needs_review":
        return { icon: "🟡", label: "Needs Review", color: "yellow" };
      case "verified":
        return { icon: "🟢", label: "Checked", color: "green" };
    }
  };


  if (isLoading) {
    return (
      <Flex justify="center" align="center" p="8">
        <Text>Loading facts...</Text>
      </Flex>
    );
  }

  if (isError) {
    return (
      <Flex justify="center" align="center" p="8" direction="column" gap="4">
        <Text color="red">Error loading facts</Text>
        <Button onClick={() => refetch()}>Retry</Button>
      </Flex>
    );
  }

  return (
    <Flex
      direction="column"
      gap="3"
      p="24px"
      style={{ height: "100%", display: "flex", overflow: "hidden" }}
      className="fact-review-container"
    >
      <Flex direction="column" gap="3" mb="3" style={{ flexShrink: 0 }}>
        <Flex direction="column" gap="1">
          <Text size="6" weight="medium">
            Verify extracted data to analyze
          </Text>
        </Flex>
        {factsList.length > 0 && (
          <Flex justify="end" align="center" gap="12px">
            {hasLockedFacts && (
              <Flex
                onClick={() => !resetMutation.isLoading && resetMutation.mutate()}
                align="center"
                gap="6px"
                style={{
                  cursor: resetMutation.isLoading ? "not-allowed" : "pointer",
                  opacity: resetMutation.isLoading ? 0.5 : 1,
                  padding: "4px 8px",
                  borderRadius: "4px",
                  transition: "background-color 0.2s",
                }}
                onMouseEnter={(e) => {
                  if (!resetMutation.isLoading) {
                    e.currentTarget.style.backgroundColor = "#f0f0f0";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "transparent";
                }}
              >
                <Text size="3">🔄</Text>
                <Text size="2" style={{ color: "#e74c3c", fontWeight: "500" }}>
                  Reset All
                </Text>
              </Flex>
            )}
            {factsList.length > 0 && (
              <>
                {selectedFacts.size > 0 && (
                  <Flex
                    onClick={() => !approveMutation.isLoading && handleApproveSelected()}
                    align="center"
                    gap="6px"
                    style={{
                      cursor: approveMutation.isLoading ? "not-allowed" : "pointer",
                      opacity: approveMutation.isLoading ? 0.5 : 1,
                      padding: "4px 8px",
                      borderRadius: "4px",
                      transition: "background-color 0.2s",
                    }}
                    onMouseEnter={(e) => {
                      if (!approveMutation.isLoading) {
                        e.currentTarget.style.backgroundColor = "#f0f0f0";
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "transparent";
                    }}
                  >
                    <Text size="3">✓</Text>
                    <Text size="2" style={{ color: "#1976D2", fontWeight: "500" }}>
                      Verify Selected ({selectedFacts.size})
                    </Text>
                  </Flex>
                )}
                <Flex
                  onClick={() => !approveMutation.isLoading && handleVerifyAll()}
                  align="center"
                  gap="6px"
                  style={{
                    cursor: approveMutation.isLoading ? "not-allowed" : "pointer",
                    opacity: approveMutation.isLoading ? 0.5 : 1,
                    padding: "4px 8px",
                    borderRadius: "4px",
                    transition: "background-color 0.2s",
                  }}
                  onMouseEnter={(e) => {
                    if (!approveMutation.isLoading) {
                      e.currentTarget.style.backgroundColor = "#f0f0f0";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  <Text size="3">✓</Text>
                  <Text size="2" style={{ color: "#1976D2", fontWeight: "500" }}>
                    Verify All
                  </Text>
                </Flex>
              </>
            )}
            <Button
              size="2"
              onClick={handleRunUnderwriting}
              style={{
                backgroundColor: "#111",
                color: "#fff",
              }}
            >
              Run Analysis →
            </Button>
          </Flex>
        )}
      </Flex>

      {/* Add Fact Form */}
      <Card
        style={{
          padding: "16px",
          marginBottom: "16px",
          backgroundColor: "#f9fafb",
          border: "1px solid #e0e0e0",
        }}
      >
        <Flex direction="column" gap="12px">
          <Flex justify="between" align="center">
            <Text size="3" weight="bold">
              Add New Fact
            </Text>
            {!showAddFactForm && (
              <button
                onClick={() => setShowAddFactForm(true)}
                style={{
                  backgroundColor: "transparent",
                  border: "1px solid #1976D2",
                  color: "#000",
                  padding: "8px 16px",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                + Add Fact
              </button>
            )}
          </Flex>

          {showAddFactForm && (
            <Flex direction="column" gap="12px">
              <Flex gap="8px" align="center" wrap="wrap">
                <TextField.Root
                  placeholder="Field name (e.g., Gross Rent)"
                  value={newFactLabel}
                  onChange={(e) => setNewFactLabel(e.target.value)}
                  style={{ flex: "1 1 200px", minWidth: "200px" }}
                />
                <TextField.Root
                  placeholder="Value"
                  value={newFactValue}
                  onChange={(e) => setNewFactValue(e.target.value)}
                  style={{ flex: "1 1 150px", minWidth: "150px" }}
                />
                <TextField.Root
                  placeholder="Unit (optional)"
                  value={newFactUnit}
                  onChange={(e) => setNewFactUnit(e.target.value)}
                  style={{ width: "120px" }}
                />
                <Button
                  size="2"
                  onClick={handleAddFact}
                  disabled={createFactMutation.isLoading || !newFactLabel.trim() || !newFactValue.trim()}
                  style={{ cursor: "pointer" }}
                >
                  {createFactMutation.isLoading ? "Adding..." : "Add"}
                </Button>
                <Button
                  size="2"
                  variant="soft"
                  onClick={() => {
                    setShowAddFactForm(false);
                    setNewFactLabel("");
                    setNewFactValue("");
                    setNewFactUnit("");
                  }}
                  style={{ cursor: "pointer" }}
                >
                  Cancel
                </Button>
              </Flex>
            </Flex>
          )}

          {/* Seeded Fields Quick Add */}
          {missingSeededFields.length > 0 && (
            <Flex direction="column" gap="8px" style={{ marginTop: "8px" }}>
              <Text size="2" style={{ color: "#666" }}>
                Quick Add:
              </Text>
              <Flex gap="8px" wrap="wrap">
                {missingSeededFields.map((field) => (
                  <button
                    key={field.label}
                    onClick={() => handleAddSeededField(field)}
                    style={{
                      backgroundColor: "transparent",
                      border: "1px solid #1976D2",
                      color: "#000",
                      padding: "4px 12px",
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontSize: "12px",
                      fontWeight: "500",
                    }}
                  >
                    + {field.label}
                  </button>
                ))}
              </Flex>
            </Flex>
          )}
        </Flex>
      </Card>

      {factsList.length === 0 ? (
        <Flex
          direction="column"
          align="center"
          justify="center"
          p="24px"
          style={{ flex: 1 }}
        >
          <Text size="3" style={{ color: "#666", marginBottom: "16px" }}>
            No facts extracted yet. Add facts manually using the form above.
          </Text>
        </Flex>
      ) : (
        <Flex direction="column" gap="2" style={{ flex: 1, overflowY: "auto", minHeight: 0, paddingBottom: "16px" }}>
          {factsList.map((fact) => {
          const status = getFactStatus(fact);
          const statusDisplay = getStatusDisplay(status);
          const isEdited = !!editedFacts[fact.fact_id];

          return (
            <Card
              key={fact.fact_id}
              className="fact-card"
              style={{
                padding: "12px 16px",
                background: fact.locked ? "#f8f9fa" : "white",
                border: selectedFacts.has(fact.fact_id)
                  ? "2px solid #111"
                  : "1px solid #e0e0e0",
              }}
            >
              <Flex direction="column" gap="2">
                <Flex justify="between" align="center" wrap="wrap" gap="2">
                  <Flex align="center" gap="2" style={{ flex: "1 1 200px", minWidth: 0 }}>
                    {!fact.locked && (
                      <input
                        type="checkbox"
                        checked={selectedFacts.has(fact.fact_id)}
                        onChange={() => toggleFactSelection(fact.fact_id)}
                        style={{ cursor: "pointer", flexShrink: 0 }}
                      />
                    )}
                    <Text size="3" weight="bold" style={{ flexShrink: 0 }}>
                      {fact.label}
                    </Text>
                    <Badge color={statusDisplay.color as any} style={{ flexShrink: 0 }}>
                      {statusDisplay.icon} {statusDisplay.label}
                    </Badge>
                    {fact.locked && (
                      <Badge color="green" style={{ flexShrink: 0 }}>
                        Locked
                      </Badge>
                    )}
                  </Flex>
                  <Flex direction="column" align="end" gap="1" style={{ fontSize: "11px", flexShrink: 0 }}>
                    <Text size="1" color="gray" style={{ textAlign: "right" }}>
                      <strong>Source:</strong> {fact.source_citation.document}, Page{" "}
                      {fact.source_citation.page}
                      {fact.source_citation.line && ` - Line: ${fact.source_citation.line}`}
                    </Text>
                    {fact.approved_at && (
                      <Text size="1" color="gray" style={{ textAlign: "right" }}>
                        <strong>Verified:</strong>{" "}
                        {new Date(fact.approved_at).toLocaleDateString()}
                        {fact.approved_by && ` by ${fact.approved_by}`}
                      </Text>
                    )}
                  </Flex>
                </Flex>

                <Flex gap="2" align="center" wrap="wrap">
                  <TextField.Root
                    value={
                      editedFacts[fact.fact_id]?.value ?? fact.value
                    }
                    onChange={(e) =>
                      handleValueChange(fact.fact_id, e.target.value)
                    }
                    disabled={fact.locked}
                    style={{ width: "200px", flexShrink: 0 }}
                    placeholder="Value"
                  />
                  {fact.unit && (
                    <TextField.Root
                      value={
                        editedFacts[fact.fact_id]?.unit ?? fact.unit
                      }
                      onChange={(e) =>
                        handleUnitChange(fact.fact_id, e.target.value)
                      }
                      disabled={fact.locked}
                      style={{ width: "120px", flexShrink: 0 }}
                      placeholder="Unit"
                    />
                  )}
                  {isEdited && !fact.locked && (
                    <Button
                      size="2"
                      onClick={() => saveFact(fact.fact_id)}
                      disabled={updateMutation.isLoading}
                      style={{ flexShrink: 0 }}
                    >
                      Save
                    </Button>
                  )}
                </Flex>
              </Flex>
            </Card>
          );
        })}
        </Flex>
      )}
    </Flex>
  );
};

export default FactReviewDeal;
