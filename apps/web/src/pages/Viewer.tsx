import { Flex, ScrollArea } from "@radix-ui/themes";
import { SegmentChunk } from "../components/SegmentChunk/SegmentChunk";
import { PDF } from "../components/PDF/PDF";
import Header from "../components/Header/Header";

export const Viewer = () => {
  return (
    <Flex direction="column" width="100%">
      <Flex
        width="100%"
        direction="column"
        style={{ boxShadow: "0px 12px 12px 0px rgba(0, 0, 0, 0.12)" }}
      >
        <Header py="24px" px="24px" />
      </Flex>
      <Flex
        direction="row"
        width="100%"
        style={{ borderTop: "2px solid var(--cyan-12)" }}
      >
        <Flex width="100%" direction="column">
          <PDF />
        </Flex>
        <ScrollArea
          scrollbars="vertical"
          type="always"
          style={{
            height: "calc(100vh - 90px)",
          }}
        >
          <Flex width="100%" height="100%" direction="column" p="24px" gap="7">
            {[0, 1, 2, 3, 4].map(() => {
              return (
                <SegmentChunk
                  segment={{
                    left: "HI",
                    top: "HI",
                    width: "HI",
                    height: "HI",
                    page_number: 0,
                    page_width: "HI",
                    page_height: "HI",
                    text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum",
                    type: "Caption",
                  }}
                />
              );
            })}
          </Flex>
        </ScrollArea>
      </Flex>
    </Flex>
  );
};

{
  /* <div className="flex border-t border-cyan-6 w-full justify-between">
<div className="border-l w-full h-full">
  <PDF />
</div>
<div className="border-l w-full p-4 space-y-4">
  {[0, 1, 2, 3, 4].map(() => {
    return (
      <SegmentChunk
        segment={{
          left: "HI",
          top: "HI",
          width: "HI",
          height: "HI",
          page_number: 0,
          page_width: "HI",
          page_height: "HI",
          text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum",
          type: "Caption",
        }}
      />
    );
  })}
</div>
</div> */
}
