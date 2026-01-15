import { useState } from "react";
import { Flex, Text, Button, TextField, Badge, ScrollArea } from "@radix-ui/themes";
import { useQuery } from "react-query";
import { getDeals, DealResponse } from "../../services/dealApi";
import { getAllContacts, getContactsByType, getFamilyOfficeContacts } from "../../services/contactApi";
import ImportContactsModal from "../Contacts/ImportContactsModal";
import "./LeftNavPane.css";

interface LeftNavPaneProps {
  selectedDealId: string | null;
  onSelectDeal: (dealId: string) => void;
  onNewDeal: () => void;
  selectedContactType?: string | null;
  onSelectContactType?: (type: string | null) => void;
}

export default function LeftNavPane({
  selectedDealId,
  onSelectDeal,
  onNewDeal,
  selectedContactType,
  onSelectContactType,
}: LeftNavPaneProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showImportModal, setShowImportModal] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({
    active: true,
    in_review: false,
    completed: false,
    shared: false,
  });
  const [expandedContacts, setExpandedContacts] = useState<Record<string, boolean>>({
    all: true,
    investors: false,
    institutional: false,
  });

  const { data: deals = [] } = useQuery<DealResponse[]>("deals", getDeals);
  
  // Get contact counts
  const allContacts = getAllContacts();
  const investorContacts = getContactsByType("investor");
  const institutionalContacts = getContactsByType("institutional");
  const familyOfficeContacts = getFamilyOfficeContacts();

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  const toggleContactGroup = (group: string) => {
    setExpandedContacts((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  const groupDeals = () => {
    const filtered = deals.filter((deal) =>
      deal.deal_name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return {
      active: filtered.filter(
        (d) =>
          d.status === "processing_documents" ||
          d.status === "extracting_facts" ||
          d.status === "pending_review"
      ),
      in_review: filtered.filter((d) => d.status === "ready_for_underwriting"),
      completed: filtered.filter((d) => d.status === "completed"),
      shared: filtered.filter((d) => d.status === "shared"),
    };
  };

  const groupedDeals = groupDeals();

  const getStatusBadge = (status: string) => {
    const badges: Record<string, { color: any; label: string }> = {
      processing_documents: { color: "blue", label: "Processing" },
      extracting_facts: { color: "orange", label: "Extracting" },
      pending_review: { color: "yellow", label: "Review" },
      ready_for_underwriting: { color: "yellow", label: "Ready" },
      completed: { color: "green", label: "Complete" },
      shared: { color: "purple", label: "Shared" },
    };
    return badges[status] || { color: "gray", label: status };
  };

  const renderDealItem = (deal: DealResponse) => {
    const badge = getStatusBadge(deal.status);
    const isSelected = deal.deal_id === selectedDealId;

    return (
      <Flex
        key={deal.deal_id}
        direction="column"
        p="12px"
        style={{
          cursor: "pointer",
          borderBottom: "1px solid #e0e0e0",
          background: isSelected ? "#f0f0f0" : "transparent",
          transition: "background 0.2s",
        }}
        className="deal-item"
        onClick={() => onSelectDeal(deal.deal_id)}
      >
        <Text size="2" weight="medium" style={{ marginBottom: "6px" }}>
          {deal.deal_name}
        </Text>
        <Flex gap="8px" align="center">
          <Badge color={badge.color} size="1">
            {badge.label}
          </Badge>
          <Text size="1" style={{ color: "#666" }}>
            {deal.document_count || 0} docs
          </Text>
        </Flex>
      </Flex>
    );
  };

  const renderGroup = (
    title: string,
    icon: string,
    groupKey: string,
    deals: DealResponse[]
  ) => {
    const isExpanded = expandedGroups[groupKey];

    return (
      <Flex direction="column" style={{ marginBottom: "8px" }}>
        <Flex
          align="center"
          justify="between"
          p="8px 12px"
          style={{
            cursor: "pointer",
            backgroundColor: "#f8f9fa",
          }}
          onClick={() => toggleGroup(groupKey)}
        >
          <Flex align="center" gap="8px">
            <Text size="2">{isExpanded ? "▼" : "▶"}</Text>
            <Text size="2" weight="medium">
              {icon} {title}
            </Text>
            <Badge size="1" variant="soft">
              {deals.length}
            </Badge>
          </Flex>
        </Flex>
        {isExpanded && deals.map((deal) => renderDealItem(deal))}
      </Flex>
    );
  };

  return (
    <Flex
      direction="column"
      style={{
        width: "260px",
        height: "100vh",
        borderRight: "1px solid #e0e0e0",
        backgroundColor: "#fff",
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <Flex
        direction="column"
        p="16px"
        style={{
          borderBottom: "1px solid #e0e0e0",
        }}
      >
        <Flex align="center" gap="12px" style={{ marginBottom: "16px" }}>
          <img
            src="/logo-orin.png"
            alt="Orin Logo"
            width={32}
            height={32}
          />
          <Text size="5" weight="bold" style={{ color: "#111" }}>
            Orin
          </Text>
        </Flex>

        {/* Search */}
        <TextField.Root
          placeholder="Search deals..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          size="2"
        />
      </Flex>

      {/* Deals List */}
      <ScrollArea
        style={{ flex: 1 }}
        scrollbars="vertical"
      >
        <Flex direction="column" p="8px 0">
          <Text
            size="2"
            weight="bold"
            style={{
              color: "#666",
              textTransform: "uppercase",
              padding: "12px 16px",
              letterSpacing: "0.5px",
            }}
          >
            My Deals
          </Text>

          {renderGroup("New", "🔵", "active", groupedDeals.active)}
          {renderGroup("Underwriting", "🟡", "in_review", groupedDeals.in_review)}
          {renderGroup("Ready to Fund", "🟢", "completed", groupedDeals.completed)}
          {renderGroup("Closed", "🟣", "shared", groupedDeals.shared)}

          {/* MY CONTACTS Section */}
          <Text
            size="2"
            weight="bold"
            style={{
              color: "#666",
              textTransform: "uppercase",
              padding: "16px 12px 8px",
              letterSpacing: "0.5px",
            }}
          >
            My Contacts
          </Text>

          {/* All Contacts */}
          <Flex
            direction="column"
            style={{ marginBottom: "4px" }}
          >
            <Flex
              align="center"
              justify="between"
              p="8px 12px"
              style={{
                cursor: "pointer",
                backgroundColor: selectedContactType === "all" ? "#f0f0f0" : "transparent",
              }}
              onClick={() => onSelectContactType && onSelectContactType("all")}
            >
              <Text size="2">All Contacts</Text>
              <Badge size="1" variant="soft">
                {allContacts.length}
              </Badge>
            </Flex>
          </Flex>

          {/* Accredited Investors */}
          <Flex
            direction="column"
            style={{ marginBottom: "4px" }}
          >
            <Flex
              align="center"
              justify="between"
              p="8px 12px"
              style={{
                cursor: "pointer",
                backgroundColor: selectedContactType === "investors" ? "#f0f0f0" : "transparent",
              }}
              onClick={() => onSelectContactType && onSelectContactType("investors")}
            >
              <Text size="2">Accredited Investors</Text>
              <Badge size="1" variant="soft">
                {investorContacts.length}
              </Badge>
            </Flex>
          </Flex>

          {/* Institutional */}
          <Flex direction="column" style={{ marginBottom: "8px" }}>
            <Flex
              align="center"
              justify="between"
              p="8px 12px"
              style={{
                cursor: "pointer",
                backgroundColor: expandedContacts.institutional ? "#f8f9fa" : "transparent",
              }}
              onClick={() => toggleContactGroup("institutional")}
            >
              <Flex align="center" gap="8px">
                <Text size="2">{expandedContacts.institutional ? "▼" : "▶"}</Text>
                <Text size="2">Institutional</Text>
                <Badge size="1" variant="soft">
                  {institutionalContacts.length}
                </Badge>
              </Flex>
            </Flex>
            {expandedContacts.institutional && (
              <Flex
                align="center"
                justify="between"
                p="8px 12px 8px 28px"
                style={{
                  cursor: "pointer",
                  backgroundColor: selectedContactType === "family_office" ? "#f0f0f0" : "transparent",
                }}
                onClick={() => onSelectContactType && onSelectContactType("family_office")}
              >
                <Text size="2">Family Offices</Text>
                <Badge size="1" variant="soft">
                  {familyOfficeContacts.length}
                </Badge>
              </Flex>
            )}
          </Flex>

          {/* Import Contacts Button */}
          <Flex 
            p="9px 12px"
            align="center"
            gap="6px"
            onClick={() => setShowImportModal(true)}
            style={{
              cursor: "pointer",
              color: "#666",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <Text size="2" weight="bold" style={{ color: "#666" }}>Import Contacts</Text>
          </Flex>

          <Text
            size="2"
            weight="bold"
            style={{
              color: "#666",
              textTransform: "uppercase",
              padding: "16px 12px 8px",
              letterSpacing: "0.5px",
            }}
          >
            Shared Packages
          </Text>
          <Text
            size="2"
            style={{
              color: "#999",
              padding: "8px 12px",
              fontStyle: "italic",
            }}
          >
            No shared packages yet
          </Text>
        </Flex>
      </ScrollArea>

      {/* New Deal Button */}
      <Flex
        p="16px"
        style={{
          borderTop: "1px solid #e0e0e0",
        }}
      >
        <Button
          size="3"
          onClick={onNewDeal}
          style={{
            width: "100%",
            backgroundColor: "#1976D2",
            color: "#fff",
            cursor: "pointer",
          }}
        >
          + New Deal
        </Button>
      </Flex>

      {/* Import Contacts Modal */}
      <ImportContactsModal
        open={showImportModal}
        onClose={() => setShowImportModal(false)}
        onImportComplete={() => {
          setShowImportModal(false);
          // Refetch contacts by invalidating query if using react-query
        }}
      />
    </Flex>
  );
}

