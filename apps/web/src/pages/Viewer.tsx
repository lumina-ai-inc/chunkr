import { Text } from "@radix-ui/themes";
import BetterButton from "../components/BetterButton";
import { useState } from "react";

export const Viewer = () => {

  return (
    <div className="flex border-t border-cyan-6 w-full justify-between">
      <div className="border-l w-full h-full">
        <Text>I'm pdf bro</Text>
      </div>
      <div className="border-l w-full p-4 space-y-4">
        {[0, 1, 2, 3, 4].map(() => {

          return (
            <Chunk />
          );
        })}
      </div>
    </div>);
};

export const Chunk = () => {

  const [markdownSelected, markdownNotSelected] = useState<boolean>(true);

  const selectedColor: any = {
    "background-color": "var(--cyan-9)",
  };

  return (
    <div className="border-2">
      <div className="flex h-16 border-b-2 justify-between px-6 items-center">
        <Text size="6" className="cyan-3"> Header 2x </Text>
        <div className="flex space-x-6">
          <BetterButton>
            <Text>Markdown </Text>
          </BetterButton>
          <BetterButton>
            <Text>JSON</Text>
          </BetterButton>
        </div>
      </div>
      <div className="px-3 py-2">
        <Text className="font-cyan-6 text-[#e5e7eb]">
          Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        </Text>
      </div>
    </div>
  );
}
