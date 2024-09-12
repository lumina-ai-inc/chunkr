import { Flex, Text, Separator } from "@radix-ui/themes";
import "./Footer.css";

export default function Footer() {
  return (
    <Flex className="footer-wrapper">
      <Flex direction="column" className="footer-container">
        <Flex direction="row" justify="between" className="footer-content">
          <Flex direction="column" gap="3">
            <Flex direction="row" gap="3" align="center" ml="-10px">
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  height: "32px",
                  width: "32px",
                }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="32"
                  height="32"
                  viewBox="0 0 32 32"
                  fill="none"
                >
                  <path
                    d="M7.84 13.04C8.56691 13.331 9.36323 13.4022 10.1302 13.2449C10.8973 13.0875 11.6012 12.7085 12.1549 12.1549C12.7085 11.6012 13.0875 10.8973 13.2449 10.1302C13.4022 9.36323 13.331 8.56691 13.04 7.84C13.8857 7.61229 14.6334 7.11328 15.1681 6.41958C15.7028 5.72588 15.9951 4.87585 16 4C18.3734 4 20.6935 4.70379 22.6668 6.02236C24.6402 7.34094 26.1783 9.21509 27.0866 11.4078C27.9948 13.6005 28.2324 16.0133 27.7694 18.3411C27.3064 20.6689 26.1635 22.8071 24.4853 24.4853C22.8071 26.1635 20.6689 27.3064 18.3411 27.7694C16.0133 28.2324 13.6005 27.9948 11.4078 27.0866C9.21509 26.1783 7.34094 24.6402 6.02236 22.6668C4.70379 20.6935 4 18.3734 4 16C4.87585 15.9951 5.72588 15.7028 6.41958 15.1681C7.11328 14.6334 7.61229 13.8857 7.84 13.04Z"
                    stroke="white"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
              </div>
              <Text size="6" weight="bold" className="cyan-1">
                chunkr
              </Text>
            </Flex>
          </Flex>
          <Flex direction="row" gap="8" className="footer-links">
            <FooterColumn
              title="Product"
              links={["Pricing", "API Docs", "Github"]}
            />
            <FooterColumn title="Contact" links={["Email", "Discord"]} />
          </Flex>
        </Flex>
        <Separator size="4" className="footer-separator" />
        <Flex
          direction="row"
          justify="between"
          align="center"
          className="footer-bottom"
        >
          <Text size="1" style={{ color: "hsla(0, 0%, 100%, 0.8)" }}>
            Chunk My Docs is maintained by Lumina AI Inc.
          </Text>
          <Flex direction="row" gap="5" align="center">
            <a
              href="https://twitter.com/lumina_ai_inc"
              target="_blank"
              rel="noopener noreferrer"
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"
                  fill="#FFF"
                />
              </svg>
            </a>
            <a
              href="https://github.com/lumina-ai-inc/chunk-my-docs"
              target="_blank"
              rel="noopener noreferrer"
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M12 0C5.37 0 0 5.37 0 12C0 17.31 3.435 21.795 8.205 23.385C8.805 23.49 9.03 23.13 9.03 22.815C9.03 22.53 9.015 21.585 9.015 20.58C6 21.135 5.22 19.845 4.98 19.17C4.845 18.825 4.26 17.76 3.75 17.475C3.33 17.25 2.73 16.695 3.735 16.68C4.68 16.665 5.355 17.55 5.58 17.91C6.66 19.725 8.385 19.215 9.075 18.9C9.18 18.12 9.495 17.595 9.84 17.295C7.17 16.995 4.38 15.96 4.38 11.37C4.38 10.065 4.845 8.985 5.61 8.145C5.49 7.845 5.07 6.615 5.73 4.965C5.73 4.965 6.735 4.65 9.03 6.195C9.99 5.925 11.01 5.79 12.03 5.79C13.05 5.79 14.07 5.925 15.03 6.195C17.325 4.635 18.33 4.965 18.33 4.965C18.99 6.615 18.57 7.845 18.45 8.145C19.215 8.985 19.68 10.05 19.68 11.37C19.68 15.975 16.875 16.995 14.205 17.295C14.64 17.67 15.015 18.39 15.015 19.515C15.015 21.12 15 22.41 15 22.815C15 23.13 15.225 23.505 15.825 23.385C18.2072 22.5807 20.2772 21.0497 21.7437 19.0074C23.2101 16.965 23.9993 14.5143 24 12C24 5.37 18.63 0 12 0Z"
                  fill="#FFF"
                />
              </svg>
            </a>
            <a
              href="https://discord.gg/your-discord-invite"
              target="_blank"
              rel="noopener noreferrer"
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M20.317 4.3698a19.7913 19.7913 0 00-4.885198a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z"
                  fill="#FFF"
                />
              </svg>
            </a>
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
}

const FooterColumn = ({ title, links }: { title: string; links: string[] }) => {
  return (
    <Flex direction="column" gap="2">
      <Text size="2" weight="bold" className="white">
        {title}
      </Text>
      {links.map((link, index) => (
        <Text key={index} size="2" className="white footer-link">
          {link}
        </Text>
      ))}
    </Flex>
  );
};
