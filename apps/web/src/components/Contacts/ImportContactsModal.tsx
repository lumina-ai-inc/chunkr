import { useState, useRef } from "react";
import { Flex, Text, Button, Dialog, Table, Select } from "@radix-ui/themes";
import { importContactsFromCSV } from "../../services/contactApi";
import { toast } from "react-hot-toast";

interface ImportContactsModalProps {
  open: boolean;
  onClose: () => void;
  onImportComplete: () => void;
}

interface CSVRow {
  [key: string]: string;
}

export default function ImportContactsModal({
  open,
  onClose,
  onImportComplete,
}: ImportContactsModalProps) {
  const [csvData, setCsvData] = useState<CSVRow[]>([]);
  const [columnMapping, setColumnMapping] = useState<Record<string, string>>({
    first_name: "",
    last_name: "",
    email: "",
    type: "",
  });
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const parseCSV = (text: string): CSVRow[] => {
    const lines = text.split("\n").filter((line) => line.trim());
    if (lines.length === 0) return [];

    // Detect delimiter (comma or semicolon)
    const firstLine = lines[0];
    const delimiter = firstLine.includes(",") ? "," : firstLine.includes(";") ? ";" : ",";

    // Parse header
    const headers = lines[0]
      .split(delimiter)
      .map((h) => h.trim().replace(/^"|"$/g, ""));

    // Parse rows
    const rows: CSVRow[] = [];
    for (let i = 1; i < Math.min(lines.length, 6); i++) {
      // Only parse first 5 data rows for preview
      const values = lines[i]
        .split(delimiter)
        .map((v) => v.trim().replace(/^"|"$/g, ""));
      
      const row: CSVRow = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || "";
      });
      rows.push(row);
    }

    return rows;
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const parsed = parseCSV(text);
      
      if (parsed.length === 0) {
        toast.error("Could not parse CSV file. Please check the format.");
        return;
      }

      setCsvData(parsed);
      const columns = Object.keys(parsed[0] || {});
      setAvailableColumns(columns);

      // Auto-detect column mapping
      const autoMapping: Record<string, string> = {
        first_name: "",
        last_name: "",
        email: "",
        type: "",
      };

      columns.forEach((col) => {
        const lower = col.toLowerCase();
        if (lower.includes("first") && lower.includes("name")) {
          autoMapping.first_name = col;
        } else if (lower.includes("last") && lower.includes("name")) {
          autoMapping.last_name = col;
        } else if (lower.includes("email")) {
          autoMapping.email = col;
        } else if (lower.includes("type") || lower.includes("category")) {
          autoMapping.type = col;
        }
      });

      setColumnMapping(autoMapping);
    };

    reader.readAsText(file);
  };

  const handleImport = () => {
    if (!columnMapping.email) {
      toast.error("Please map the email column");
      return;
    }

    if (!columnMapping.first_name && !columnMapping.last_name) {
      toast.error("Please map at least first name or last name");
      return;
    }

    try {
      // Re-parse full CSV
      const file = fileInputRef.current?.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const lines = text.split("\n").filter((line) => line.trim());
        const delimiter = lines[0].includes(",") ? "," : lines[0].includes(";") ? ";" : ",";
        const headers = lines[0].split(delimiter).map((h) => h.trim().replace(/^"|"$/g, ""));
        
        const allRows: CSVRow[] = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(delimiter).map((v) => v.trim().replace(/^"|"$/g, ""));
          const row: CSVRow = {};
          headers.forEach((header, index) => {
            row[header] = values[index] || "";
          });
          allRows.push(row);
        }

        const imported = importContactsFromCSV(allRows, columnMapping);
        toast.success(`Successfully imported ${imported.length} contacts`);
        onImportComplete();
        handleClose();
      };
      reader.readAsText(file);
    } catch (error) {
      toast.error("Failed to import contacts");
      console.error(error);
    }
  };

  const handleClose = () => {
    setCsvData([]);
    setColumnMapping({
      first_name: "",
      last_name: "",
      email: "",
      type: "",
    });
    setAvailableColumns([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClose();
  };

  return (
    <Dialog.Root open={open} onOpenChange={handleClose}>
      <Dialog.Content style={{ maxWidth: "700px", maxHeight: "80vh" }}>
        <Dialog.Title>Import Contacts</Dialog.Title>
        <Dialog.Description size="2" mb="4">
          Upload a CSV file with contact information (first name, last name, email, optional type)
        </Dialog.Description>

        <Flex direction="column" gap="4">
          {/* File Upload */}
          <Flex direction="column" gap="2">
            <Text size="2" weight="bold">
              Upload CSV File
            </Text>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              style={{
                padding: "8px",
                border: "1px solid #ddd",
                borderRadius: "4px",
              }}
            />
          </Flex>

          {/* Column Mapping */}
          {availableColumns.length > 0 && (
            <Flex direction="column" gap="3">
              <Text size="2" weight="bold">
                Map Columns
              </Text>
              
              <Flex direction="column" gap="2">
                <Flex align="center" gap="2">
                  <Text size="2" style={{ width: "120px" }}>
                    First Name:
                  </Text>
                  <Select.Root
                    value={columnMapping.first_name || undefined}
                    onValueChange={(value) =>
                      setColumnMapping({ ...columnMapping, first_name: value })
                    }
                  >
                    <Select.Trigger style={{ flex: 1 }} placeholder="Select column..." />
                    <Select.Content>
                      {availableColumns.map((col) => (
                        <Select.Item key={col} value={col}>
                          {col}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>
                </Flex>

                <Flex align="center" gap="2">
                  <Text size="2" style={{ width: "120px" }}>
                    Last Name:
                  </Text>
                  <Select.Root
                    value={columnMapping.last_name || undefined}
                    onValueChange={(value) =>
                      setColumnMapping({ ...columnMapping, last_name: value })
                    }
                  >
                    <Select.Trigger style={{ flex: 1 }} placeholder="Select column..." />
                    <Select.Content>
                      {availableColumns.map((col) => (
                        <Select.Item key={col} value={col}>
                          {col}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>
                </Flex>

                <Flex align="center" gap="2">
                  <Text size="2" style={{ width: "120px" }}>
                    Email: <Text style={{ color: "red" }}>*</Text>
                  </Text>
                  <Select.Root
                    value={columnMapping.email || undefined}
                    onValueChange={(value) =>
                      setColumnMapping({ ...columnMapping, email: value })
                    }
                  >
                    <Select.Trigger style={{ flex: 1 }} placeholder="Select column... *" />
                    <Select.Content>
                      {availableColumns.map((col) => (
                        <Select.Item key={col} value={col}>
                          {col}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>
                </Flex>

                <Flex align="center" gap="2">
                  <Text size="2" style={{ width: "120px" }}>
                    Type (optional):
                  </Text>
                  <Select.Root
                    value={columnMapping.type || undefined}
                    onValueChange={(value) =>
                      setColumnMapping({ ...columnMapping, type: value })
                    }
                  >
                    <Select.Trigger style={{ flex: 1 }} placeholder="Select column... (optional)" />
                    <Select.Content>
                      {availableColumns.map((col) => (
                        <Select.Item key={col} value={col}>
                          {col}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>
                </Flex>
              </Flex>
            </Flex>
          )}

          {/* Preview */}
          {csvData.length > 0 && (
            <Flex direction="column" gap="2">
              <Text size="2" weight="bold">
                Preview (first 5 rows)
              </Text>
              <div style={{ maxHeight: "200px", overflow: "auto", border: "1px solid #ddd", borderRadius: "4px" }}>
                <Table.Root>
                  <Table.Header>
                    <Table.Row>
                      {availableColumns.map((col) => (
                        <Table.ColumnHeaderCell key={col}>
                          {col}
                        </Table.ColumnHeaderCell>
                      ))}
                    </Table.Row>
                  </Table.Header>
                  <Table.Body>
                    {csvData.map((row, idx) => (
                      <Table.Row key={idx}>
                        {availableColumns.map((col) => (
                          <Table.Cell key={col}>{row[col] || ""}</Table.Cell>
                        ))}
                      </Table.Row>
                    ))}
                  </Table.Body>
                </Table.Root>
              </div>
            </Flex>
          )}
        </Flex>

        <Flex gap="3" mt="4" justify="end">
          <Dialog.Close>
            <Button variant="soft" color="gray">
              Cancel
            </Button>
          </Dialog.Close>
          <Button
            onClick={handleImport}
            disabled={!columnMapping.email || csvData.length === 0}
          >
            Import
          </Button>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
