import { Flex } from "@radix-ui/themes";

export default function Badge({ children }: { children: React.ReactNode }) {
  return (
    <Flex
      py="8px"
      px="12px"
      style={{ backgroundColor: "var(--cyan-4)", borderRadius: "4px" }}
    >
      {children}
    </Flex>
  );
}
