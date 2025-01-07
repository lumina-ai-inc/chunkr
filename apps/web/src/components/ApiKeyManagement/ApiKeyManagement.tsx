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

export default function ApiKeyManagement({ user }: ApiKeyManagementProps) {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>(
    user?.api_keys?.map((key: string) => ({
      id: crypto.randomUUID(),
      key,
      usage: Math.floor(Math.random() * 1000), // Replace with actual usage data
      limit: 10000,
    })) || []
  );

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
    <div className="usage-container">
      <Flex direction="column" gap="4">
        <Flex justify="between" align="center">
          <Text
            size="6"
            weight="bold"
            style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
          >
            API Keys
          </Text>
          <BetterButton onClick={createNewApiKey}>
            <Text
              size="2"
              weight="medium"
              style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
            >
              Create New Key
            </Text>
          </BetterButton>
        </Flex>

        {apiKeys.map((apiKey) => (
          <div key={apiKey.id} className="api-key-card">
            <Flex direction="row">
              <Flex direction="row" justify="between" align="center" gap="6">
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
                >
                  {apiKey.key}
                </Text>
                <Flex gap="4">
                  <BetterButton
                    onClick={() => navigator.clipboard.writeText(apiKey.key)}
                  >
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
                    >
                      Copy
                    </Text>
                  </BetterButton>
                  <BetterButton onClick={() => deleteApiKey(apiKey.id)}>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
                    >
                      Delete
                    </Text>
                  </BetterButton>
                </Flex>
              </Flex>
            </Flex>
          </div>
        ))}
      </Flex>
    </div>
  );
}
