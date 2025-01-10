import { Flex, Text } from "@radix-ui/themes";
import { useState } from "react";
import BetterButton from "../BetterButton/BetterButton";
import { User } from "../../models/user.model";
import "./ApiKeyManagement.css";

interface ApiKey {
  id: string;
  key: string;
  usage: number;
  limit: number;
}

interface ApiKeyManagementProps {
  user: User | undefined;
}

const dummyApiKeys = [
  {
    id: 1,
    name: "Production API Key",
    key: "sk_live_12345678901234567890abcdef",
    last_used: "2024-03-20 15:30:00",
  },
  {
    id: 2,
    name: "Development Key",
    key: "sk_dev_98765432109876543210zyxwv",
    last_used: "2024-03-19 09:15:00",
  },
  {
    id: 3,
    name: "Testing Environment",
    key: "sk_test_abcdefghijklmnopqrstuvwx",
    last_used: "2024-03-15 11:45:00",
  },
];

export default function ApiKeyManagement({ user }: ApiKeyManagementProps) {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>(
    user?.api_keys?.map((key: string) => ({
      id: crypto.randomUUID(),
      key,
      usage: Math.floor(Math.random() * 1000), // Replace with actual usage data
      limit: 10000,
    })) || []
  );

  const [expandedKeyId, setExpandedKeyId] = useState<number | null>(null);

  const createNewApiKey = async () => {
    // TODO: API call to create new key
    const newKey = {
      id: crypto.randomUUID(),
      key: "new-api-key-" + Math.random().toString(36).substring(7),
      usage: 0,
      limit: 10000,
    };
    setApiKeys([...apiKeys, newKey]);
  };

  const deleteApiKey = async (keyId: string) => {
    // TODO: API call to delete key
    setApiKeys(apiKeys.filter((k) => k.id !== keyId));
  };

  return (
    <Flex direction="column" className="api-key-management" gap="4">
      <Flex
        direction="row"
        justify="between"
        align="center"
        className="api-key-header"
      >
        <Text size="5" weight="bold" style={{ color: "#FFF" }}>
          Manage API Keys
        </Text>
        <BetterButton
          onClick={() => {
            /* TODO: Add create key handler */
          }}
        >
          <Text size="2" weight="medium" style={{ color: "#FFF" }}>
            Create New Key
          </Text>
        </BetterButton>
      </Flex>

      <table className="api-key-table">
        <thead>
          <tr>
            <th>
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                Name
              </Text>
            </th>
            <th>
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                Key
              </Text>
            </th>
            <th>
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                Last Used
              </Text>
            </th>
            <th>
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                Actions
              </Text>
            </th>
          </tr>
        </thead>
        <tbody>
          {dummyApiKeys.map((apiKey) => (
            <tr key={apiKey.id}>
              <td>
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "rgba(255, 255, 255, 0.8)" }}
                >
                  {apiKey.name || "API Key"}
                </Text>
              </td>
              <td>
                <div className="key-cell" title={apiKey.key}>
                  <Text
                    size="2"
                    weight="medium"
                    style={{ color: "rgba(255, 255, 255, 0.8)" }}
                  >
                    {expandedKeyId === apiKey.id
                      ? apiKey.key
                      : `${apiKey.key.slice(0, 5)}...`}
                  </Text>
                </div>
              </td>
              <td>
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "rgba(255, 255, 255, 0.8)" }}
                >
                  {apiKey.last_used || "Never"}
                </Text>
              </td>
              <td>
                <Flex gap="4">
                  <BetterButton
                    onClick={() =>
                      setExpandedKeyId(
                        expandedKeyId === apiKey.id ? null : apiKey.id
                      )
                    }
                  >
                    <Text size="1">
                      {expandedKeyId === apiKey.id ? "Hide" : "View"}
                    </Text>
                  </BetterButton>
                  <BetterButton
                    onClick={() => navigator.clipboard.writeText(apiKey.key)}
                  >
                    <Text size="1">Copy</Text>
                  </BetterButton>
                  <BetterButton onClick={() => deleteApiKey(apiKey.id)}>
                    <Text size="1">Delete</Text>
                  </BetterButton>
                </Flex>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Flex>
  );
}
