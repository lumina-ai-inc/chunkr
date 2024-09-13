import { Flex } from "@radix-ui/themes";

export default function Badge({
  children,
  className,
}: {
  children: React.ReactNode;
  className: string;
}) {
  return (
    <Flex
      py="8px"
      px="12px"
      className={className}
      style={{ borderRadius: "4px" }}
    >
      {children}
    </Flex>
  );
}
