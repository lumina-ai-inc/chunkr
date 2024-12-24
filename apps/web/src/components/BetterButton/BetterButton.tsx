import "./BetterButton.css";

export default function BetterButton({
  children,
  padding = "8px 10px",
  radius = "4px",
  active = false,
  disabled = false,
  onClick,
}: {
  children: React.ReactNode;
  padding?: string;
  radius?: string;
  active?: boolean;
  disabled?: boolean;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
}) {
  return (
    <button
      className={`button-resting ${active ? "button-active" : ""}`}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "8px",
        padding,
        width: "fit-content",
        fontSize: "12px",
        borderRadius: radius,
      }}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
