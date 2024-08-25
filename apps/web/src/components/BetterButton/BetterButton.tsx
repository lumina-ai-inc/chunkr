import "./BetterButton.css";

export default function BetterButton({
  children,
  padding = "4px 12px",
  active = false,
  onClick,
}: {
  children: React.ReactNode;
  padding?: string;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      className={`button-resting ${active ? "button-active" : ""}`}
      style={{ padding }}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
