import "./BetterButton.css";

export default function BetterButton({
  children,
  padding = "4px 10px",
  active = false,
  disabled = false,
  onClick,
}: {
  children: React.ReactNode;
  padding?: string;
  active?: boolean;
  disabled?: boolean;
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
      disabled={disabled}
    >
      {children}
    </button>
  );
}
