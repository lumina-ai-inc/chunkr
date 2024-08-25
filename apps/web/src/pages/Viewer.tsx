import { Text } from "@radix-ui/themes";

export const Viewer = () => <div className="flex border-t border-cyan-6 w-full justify-between">
  <div className="border-l w-full h-full">
    <Text>I'm pdf bro</Text>
  </div>
  <div className="border-l w-full p-4 space-y-4">
    {[0, 1, 2, 3, 4].map(() => {

      return (
        <div className="border-2">

          <div className="flex border-b-2 justify-between">
            <Text> Header 2x </Text>
            <div className="flex">
              <Text> Markdown </Text>
              <Text> Json</Text>
            </div>
          </div>

          <div className="p-40">
            <Text> What is up bro, I'ma chunk </Text>
          </div>
        </div>
      );
    })}
  </div>

</div>;
