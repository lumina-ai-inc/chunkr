import "./BetterButton.css";

export default function BetterButton({
  children,
  padding = "4px 10px",
  active = false,
  onClick,
}: {
  children: React.ReactNode;
  padding?: string;
  active?: boolean;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
}) {
  return (
    <button
      className={`button-resting ${active ? "button-active" : ""}`}
      style={{
        padding,
        width: "fit-content",
        fontSize: "12px",
      }}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
