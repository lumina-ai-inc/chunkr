import { Text } from "@radix-ui/themes";
import { SegmentChunk } from "../components/SegmentChunk/SegmentChunk";
import { PDF } from "../components/PDF/PDF";

export const Viewer = () => {
  return (
    <div className="flex border-t border-cyan-6 w-full justify-between">
      <div className="border-l w-full h-full">
        <PDF/>
      </div>
      <div className="border-l w-full p-4 space-y-4">
        {[0, 1, 2, 3, 4].map(() => {
          return (
            <SegmentChunk segment={{
              left: "HI",
              top: "HI",
              width: "HI",
              height: "HI",
              page_number: 0,
              page_width: "HI",
              page_height: "HI",
              text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum",
              type: "Caption"
            }}
            />
          );
        })}
      </div>
    </div>
  );
};
