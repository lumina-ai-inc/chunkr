// Contact management service
// Uses localStorage for now, similar to dealApi mock pattern

export interface Contact {
  contact_id: string;
  first_name: string;
  last_name: string;
  email: string;
  title?: string;
  company?: string;
  person_linkedin_url?: string;
  mobile_phone?: string;
  type?: "investor" | "institutional" | "family_office" | null;
  tags?: string[];
  created_at: string;
  updated_at: string;
}

export interface ContactGroup {
  group_id: string;
  group_name: string;
  contact_ids: string[];
  created_at: string;
}

const CONTACTS_STORAGE_KEY = "orin_contacts";
const CONTACT_GROUPS_STORAGE_KEY = "orin_contact_groups";

// Initialize with empty arrays if not exists
function getStoredContacts(): Contact[] {
  try {
    const stored = localStorage.getItem(CONTACTS_STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function saveContacts(contacts: Contact[]): void {
  localStorage.setItem(CONTACTS_STORAGE_KEY, JSON.stringify(contacts));
}

function getStoredGroups(): ContactGroup[] {
  try {
    const stored = localStorage.getItem(CONTACT_GROUPS_STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function saveGroups(groups: ContactGroup[]): void {
  localStorage.setItem(CONTACT_GROUPS_STORAGE_KEY, JSON.stringify(groups));
}

export function getAllContacts(): Contact[] {
  return getStoredContacts();
}

export function getContactsByType(type: "investor" | "institutional" | "family_office"): Contact[] {
  const contacts = getStoredContacts();
  return contacts.filter((c) => c.type === type);
}

export function getInstitutionalContacts(): Contact[] {
  const contacts = getStoredContacts();
  return contacts.filter((c) => c.type === "institutional" || c.type === "family_office");
}

export function getFamilyOfficeContacts(): Contact[] {
  const contacts = getStoredContacts();
  return contacts.filter((c) => c.type === "family_office");
}

export function importContactsFromCSV(
  csvData: Array<Record<string, string>>,
  columnMapping: Record<string, string>
): Contact[] {
  const contacts: Contact[] = [];
  const existingContacts = getStoredContacts();
  const existingEmails = new Set(existingContacts.map((c) => c.email.toLowerCase()));

  csvData.forEach((row, index) => {
    const email = (row[columnMapping.email] || "").trim().toLowerCase();
    if (!email || existingEmails.has(email)) {
      return; // Skip if no email or duplicate
    }

    const contact: Contact = {
      contact_id: `contact-${Date.now()}-${index}`,
      first_name: (row[columnMapping.first_name] || "").trim(),
      last_name: (row[columnMapping.last_name] || "").trim(),
      email: email,
      title: columnMapping.title ? (row[columnMapping.title] || "").trim() : undefined,
      company: columnMapping.company ? (row[columnMapping.company] || "").trim() : undefined,
      person_linkedin_url: columnMapping.person_linkedin_url ? (row[columnMapping.person_linkedin_url] || "").trim() : undefined,
      mobile_phone: columnMapping.mobile_phone ? (row[columnMapping.mobile_phone] || "").trim() : undefined,
      type: row[columnMapping.type]?.trim().toLowerCase() === "investor" ? "investor" :
            row[columnMapping.type]?.trim().toLowerCase() === "institutional" ? "institutional" :
            row[columnMapping.type]?.trim().toLowerCase() === "family office" ? "family_office" : null,
      tags: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    contacts.push(contact);
    existingEmails.add(email);
  });

  const allContacts = [...existingContacts, ...contacts];
  saveContacts(allContacts);
  return contacts;
}

export function updateContact(contactId: string, updates: Partial<Contact>): Contact | null {
  const contacts = getStoredContacts();
  const index = contacts.findIndex((c) => c.contact_id === contactId);
  
  if (index === -1) return null;

  contacts[index] = {
    ...contacts[index],
    ...updates,
    updated_at: new Date().toISOString(),
  };

  saveContacts(contacts);
  return contacts[index];
}

export function deleteContact(contactId: string): boolean {
  const contacts = getStoredContacts();
  const filtered = contacts.filter((c) => c.contact_id !== contactId);
  
  if (filtered.length === contacts.length) return false;
  
  saveContacts(filtered);
  return true;
}

export function getAllGroups(): ContactGroup[] {
  return getStoredGroups();
}

export function createGroup(groupName: string, contactIds: string[]): ContactGroup {
  const groups = getStoredGroups();
  const group: ContactGroup = {
    group_id: `group-${Date.now()}`,
    group_name: groupName,
    contact_ids: contactIds,
    created_at: new Date().toISOString(),
  };
  
  groups.push(group);
  saveGroups(groups);
  return group;
}

export function searchContacts(query: string): Contact[] {
  const contacts = getStoredContacts();
  const lowerQuery = query.toLowerCase();
  
  return contacts.filter(
    (c) =>
      c.first_name.toLowerCase().includes(lowerQuery) ||
      c.last_name.toLowerCase().includes(lowerQuery) ||
      c.email.toLowerCase().includes(lowerQuery)
  );
}
