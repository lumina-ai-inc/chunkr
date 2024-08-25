import styled from "styled-components";

const StyledButton = styled.button<{ padding: string }>`
  background-color: var(--cyan-12);
  border: none;
  color: rgba(222, 247, 249, 0.5);
  padding: ${(props: { padding: string }) => props.padding};
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  border-radius: 0px;
  cursor: pointer;
  transition:
    background-color 0.2s,
    color 0.2s;
  &:hover,
  &:active,
  &:focus {
    background-color: var(--cyan-9);
    color: var(--cyan-3);
    outline: none;
  }

  &:focus {
    box-shadow: none;
  }
`;

export default function BetterButton({
  children,
  padding = "4px 12px",
}: {
  children: React.ReactNode;
  padding?: string;
}) {
  return <StyledButton padding={padding}>{children}</StyledButton>;
}
