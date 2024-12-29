import { useEffect, useRef, useState } from "react";
import { Text } from "@radix-ui/themes";
import "./Dropdown.css";
interface DropdownProps {
  value: string;
  options: string[];
  onChange: (value: string) => void;
  multiple?: boolean;
  selectedValues?: string[];
}

const Dropdown = ({
  value,
  options,
  onChange,
  multiple,
  selectedValues,
}: DropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, []);

  return (
    <div className="model-selector" ref={dropdownRef}>
      <button
        className="model-selector-button"
        onClick={() => setIsOpen(!isOpen)}
      >
        {multiple ? `Selected Types (${selectedValues?.length})` : value}
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path
            d="M2.5 4.5L6 8L9.5 4.5"
            stroke="currentColor"
            strokeWidth="1.5"
          />
        </svg>
      </button>
      {isOpen && (
        <div className="model-dropdown">
          {options.map((option) => (
            <div
              key={option}
              className={`model-option ${
                multiple
                  ? selectedValues?.includes(option)
                    ? "selected"
                    : ""
                  : option === value
                    ? "selected"
                    : ""
              }`}
              onClick={() => {
                if (multiple && selectedValues) {
                  onChange(option); // Parent will handle the array logic
                } else {
                  onChange(option);
                  setIsOpen(false);
                }
              }}
            >
              <Text size="2" weight="bold">
                {option}
              </Text>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Dropdown;
