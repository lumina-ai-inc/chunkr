import { Flex } from "@radix-ui/themes";
import "./Home.css";
import Header from "../components/Header";

export const Home = () => {
  return (
    <Flex
      direction="column"
      style={{
        position: "fixed",
        height: "100%",
        width: "100%",
      }}
      className="pulsing-background"
    >
      <Header />
    </Flex>
  );
};
