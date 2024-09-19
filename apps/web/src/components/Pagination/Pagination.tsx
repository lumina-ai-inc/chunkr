import React from "react";
import { Flex, Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
}) => {
  return (
    <Flex gap="4" align="center">
      <BetterButton
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={currentPage === 1}
      >
        <Text size="1" className="white">
          Previous
        </Text>
      </BetterButton>
      <Flex
        align="center"
        justify="center"
        style={{
          padding: "6px 12px",
          borderRadius: "4px",
          backgroundColor: "hsla(180, 100%, 100%, 0.1)",
          color: "hsla(180, 100%, 100%, 0.9)",
          fontWeight: "bold",
        }}
      >
        <Text size="2" className="white">
          {currentPage} / {totalPages}
        </Text>
      </Flex>
      <BetterButton
        onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        disabled={currentPage === totalPages}
      >
        <Text size="1" className="white">
          Next
        </Text>
      </BetterButton>
    </Flex>
  );
};

export default Pagination;
