import { Text, Flex } from "@radix-ui/themes";
import "./Account.css";
import PieChartWithCenterLabel from "../Chart/Chart";

export default function Account() {
  return (
    <div className="account-module-container">
      <Flex direction="row" width="100%" justify="between">
        <Flex direction="column" gap="4px">
          <Text size="2" weight="bold" className="cyan-1">
            ACCOUNT
          </Text>
          <Text size="6" weight="bold" className="cyan-1">
            m-chadd100@gmail.com
          </Text>
        </Flex>
        <Flex direction="column" gap="4px">
          <PieChartWithCenterLabel />
        </Flex>
      </Flex>
    </div>
  );
}
