import { useState, useMemo } from "react";
import { Flex, Text, Button, Dialog, TextField, Checkbox, TextArea } from "@radix-ui/themes";
import { getAllContacts, getAllGroups, Contact, searchContacts } from "../../services/contactApi";
import { toast } from "react-hot-toast";

interface ShareWithContactsModalProps {
  open: boolean;
  onClose: () => void;
  dealName: string;
  memoData?: {
    executiveSummary?: string;
    financialHighlights?: any;
    propertyDetails?: any;
  };
}

export default function ShareWithContactsModal({
  open,
  onClose,
  dealName,
  memoData,
}: ShareWithContactsModalProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedContactIds, setSelectedContactIds] = useState<Set<string>>(new Set());
  const [selectedGroupIds, setSelectedGroupIds] = useState<Set<string>>(new Set());
  const [emailSubject, setEmailSubject] = useState(`Investment Opportunity - ${dealName}`);
  const [emailBody, setEmailBody] = useState("");

  const allContacts = getAllContacts();
  const allGroups = getAllGroups();

  // Initialize email body template
  useMemo(() => {
    if (!emailBody && memoData) {
      const template = `Dear {{first_name}},

I wanted to share an exciting investment opportunity with you: ${dealName}.

${memoData.executiveSummary || "This property presents a strong investment opportunity with solid fundamentals."}

Key Highlights:
${memoData.financialHighlights ? `
- NOI: $${memoData.financialHighlights.noi?.toLocaleString() || "N/A"}
- DSCR: ${memoData.financialHighlights.dscr || "N/A"}x
- Cap Rate: ${memoData.financialHighlights.capRate || "N/A"}%
` : ""}

I've attached the full investment memorandum for your review. Please let me know if you'd like to discuss this opportunity further.

Best regards`;
      setEmailBody(template);
    }
  }, [dealName, memoData, emailBody]);

  const filteredContacts = useMemo(() => {
    if (!searchQuery.trim()) return allContacts;
    return searchContacts(searchQuery);
  }, [searchQuery, allContacts]);

  const toggleContact = (contactId: string) => {
    const newSet = new Set(selectedContactIds);
    if (newSet.has(contactId)) {
      newSet.delete(contactId);
    } else {
      newSet.add(contactId);
    }
    setSelectedContactIds(newSet);
  };

  const toggleGroup = (groupId: string) => {
    const newSet = new Set(selectedGroupIds);
    if (newSet.has(groupId)) {
      newSet.delete(groupId);
    } else {
      newSet.add(groupId);
    }
    setSelectedGroupIds(newSet);
  };

  const getSelectedContacts = (): Contact[] => {
    const contacts: Contact[] = [];
    
    // Add individually selected contacts
    selectedContactIds.forEach((id) => {
      const contact = allContacts.find((c) => c.contact_id === id);
      if (contact) contacts.push(contact);
    });

    // Add contacts from selected groups
    selectedGroupIds.forEach((groupId) => {
      const group = allGroups.find((g) => g.group_id === groupId);
      if (group) {
        group.contact_ids.forEach((contactId) => {
          const contact = allContacts.find((c) => c.contact_id === contactId);
          if (contact && !contacts.find((c) => c.contact_id === contact.contact_id)) {
            contacts.push(contact);
          }
        });
      }
    });

    return contacts;
  };

  const handleCopyDraft = () => {
    const selected = getSelectedContacts();
    if (selected.length === 0) {
      toast.error("Please select at least one recipient");
      return;
    }

    // Create mailto link or copy to clipboard
    const recipients = selected.map((c) => c.email).join(",");
    //const mailtoLink = `mailto:${recipients}?subject=${encodeURIComponent(emailSubject)}&body=${encodeURIComponent(emailBody)}`;
    
    // Copy email content to clipboard
    const emailContent = `To: ${recipients}\nSubject: ${emailSubject}\n\n${emailBody}`;
    navigator.clipboard.writeText(emailContent).then(() => {
      toast.success("Email draft copied to clipboard!");
    });
  };

  const handleSendViaGmail = () => {
    // Phase 2: OAuth Gmail integration
    toast("Gmail integration coming soon!", { icon: "ℹ️" });
  };

  const selectedContacts = getSelectedContacts();

  return (
    <Dialog.Root open={open} onOpenChange={onClose}>
      <Dialog.Content style={{ maxWidth: "800px", maxHeight: "90vh" }}>
        <Dialog.Title>Share with Contacts</Dialog.Title>
        <Dialog.Description size="2" mb="4">
          Select recipients and customize your email message
        </Dialog.Description>

        <Flex direction="column" gap="4" style={{ maxHeight: "70vh", overflow: "auto" }}>
          {/* Recipient Selection */}
          <Flex direction="column" gap="3">
            <Text size="3" weight="bold">
              📋 Select Recipients
            </Text>

            {/* Search */}
            <TextField.Root
              placeholder="Search contacts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              size="2"
            />

            {/* Groups */}
            {allGroups.length > 0 && (
              <Flex direction="column" gap="2">
                {allGroups.map((group) => (
                  <Flex key={group.group_id} align="center" gap="2">
                    <Checkbox
                      checked={selectedGroupIds.has(group.group_id)}
                      onCheckedChange={() => toggleGroup(group.group_id)}
                    />
                    <Text size="2">
                      {group.group_name} ({group.contact_ids.length} contacts)
                    </Text>
                  </Flex>
                ))}
              </Flex>
            )}

            {/* Individual Contacts */}
            <Flex direction="column" gap="2" style={{ maxHeight: "200px", overflow: "auto" }}>
              {filteredContacts.length === 0 ? (
                <Text size="2" color="gray">
                  No contacts found
                </Text>
              ) : (
                filteredContacts.map((contact) => (
                  <Flex key={contact.contact_id} align="center" gap="2">
                    <Checkbox
                      checked={selectedContactIds.has(contact.contact_id)}
                      onCheckedChange={() => toggleContact(contact.contact_id)}
                    />
                    <Text size="2">
                      {contact.first_name} {contact.last_name} ({contact.email})
                    </Text>
                  </Flex>
                ))
              )}
            </Flex>

            {selectedContacts.length > 0 && (
              <Text size="2" color="gray">
                {selectedContacts.length} recipient(s) selected
              </Text>
            )}
          </Flex>

          {/* Email Preview */}
          <Flex direction="column" gap="3">
            <Text size="3" weight="bold">
              ✉️ Email Preview
            </Text>

            <Flex direction="column" gap="2">
              <Text size="2" weight="bold">
                Subject:
              </Text>
              <TextField.Root
                value={emailSubject}
                onChange={(e) => setEmailSubject(e.target.value)}
                size="2"
              />
            </Flex>

            <Flex direction="column" gap="2">
              <Text size="2" weight="bold">
                Body:
              </Text>
              <TextArea
                value={emailBody}
                onChange={(e) => setEmailBody(e.target.value)}
                rows={12}
                style={{ fontFamily: "monospace", fontSize: "13px" }}
              />
              <Text size="1" color="gray">
                Use {"{{first_name}}"} and {"{{last_name}}"} as merge fields
              </Text>
            </Flex>
          </Flex>
        </Flex>

        <Flex gap="3" mt="4" justify="end">
          <Dialog.Close>
            <Button variant="soft" color="gray">
              Cancel
            </Button>
          </Dialog.Close>
          <Button variant="soft" onClick={handleCopyDraft}>
            Copy Draft
          </Button>
          <Button onClick={handleSendViaGmail} disabled>
            Send via Gmail
          </Button>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
