import "./BetterButton.css";

export default function BetterButton({
  children,
  padding = "4px 12px",
  active = false,
}: {
  children: React.ReactNode;
  padding?: string;
  active?: boolean;
}) {
  return (
    <button
      className={`button-resting ${active ? "button-active" : ""}`}
      style={{ padding }}
    >
      {children}
    </button>
  );
}
