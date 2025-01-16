import { Flex, Text } from "@radix-ui/themes";
import "./ConfigControls.css";
import { useState, useEffect, useRef } from "react";
import {
  SegmentProcessing,
  GenerationStrategy,
  CroppingStrategy,
  Property,
  JsonSchema,
  DEFAULT_SEGMENT_PROCESSING,
  ChunkProcessing,
} from "../../models/taskConfig.model";

interface ToggleGroupProps {
  value: string;
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  label: React.ReactNode;
  docHover?: boolean;
}

export function ToggleGroup({
  value,
  onChange,
  options,
  label,
  docHover = true,
}: ToggleGroupProps) {
  return (
    <div className="config-card">
      <div className="config-card-header">
        <Text size="3" weight="bold" className="white">
          {label}
        </Text>
        <Flex
          direction="row"
          gap="1"
          align="center"
          justify="end"
          className={docHover ? "docs-text" : "docs-text-hidden"}
        >
          <Text size="1" weight="medium" className="white ">
            Docs
          </Text>
          <svg
            width="12px"
            height="12px"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M14.1625 18.4876L13.4417 19.2084C11.053 21.5971 7.18019 21.5971 4.79151 19.2084C2.40283 16.8198 2.40283 12.9469 4.79151 10.5583L5.51236 9.8374"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M9.8374 14.1625L14.1625 9.8374"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </Flex>
      </div>
      <div className="toggle-buttons">
        {options.map((option) => (
          <button
            key={option.value}
            className={`toggle-button ${value === option.value ? "active" : ""}`}
            onClick={() => onChange(option.value)}
          >
            <Text size="1" weight="bold">
              {option.label}
            </Text>
          </button>
        ))}
      </div>
    </div>
  );
}

interface NumberInputProps {
  value: number;
  onChange: (value: number) => void;
  label: React.ReactNode;
  min?: number;
  max?: number;
  docHover?: boolean;
}

export function NumberInput({
  value,
  onChange,
  label,
  min,
  max,
  docHover = true,
}: NumberInputProps) {
  return (
    <div className="config-card">
      <div className="config-card-header">
        <Text size="3" weight="bold" className="white">
          {label}
        </Text>
        <Flex
          direction="row"
          gap="1"
          align="center"
          justify="end"
          className={docHover ? "docs-text" : "docs-text-hidden"}
        >
          <Text size="1" weight="medium" className="white ">
            Docs
          </Text>
          <svg
            width="12px"
            height="12px"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M14.1625 18.4876L13.4417 19.2084C11.053 21.5971 7.18019 21.5971 4.79151 19.2084C2.40283 16.8198 2.40283 12.9469 4.79151 10.5583L5.51236 9.8374"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M9.8374 14.1625L14.1625 9.8374"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </Flex>
      </div>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        className="number-input"
      />
    </div>
  );
}

interface SegmentProcessingControlsProps {
  value: SegmentProcessing;
  onChange: (value: SegmentProcessing) => void;
  showOnlyPage?: boolean;
}

export function SegmentProcessingControls({
  value,
  onChange,
  showOnlyPage = false,
}: SegmentProcessingControlsProps) {
  const [selectedType, setSelectedType] =
    useState<keyof SegmentProcessing>("Text");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const segmentTypes = showOnlyPage
    ? (["Page"] as (keyof SegmentProcessing)[])
    : (Object.keys(value).filter(
        (key) => key !== "Page"
      ) as (keyof SegmentProcessing)[]);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (showOnlyPage && selectedType !== "Page") {
      setSelectedType("Page");
    } else if (!showOnlyPage && selectedType === "Page") {
      setSelectedType("Text"); // or any other default segment type
    }
  }, [selectedType, showOnlyPage]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsDropdownOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const isSegmentModified = (type: keyof SegmentProcessing) => {
    const segmentDefaults = {
      // Default config for most segments
      ...DEFAULT_SEGMENT_PROCESSING,
    };

    const defaultConfig = segmentDefaults[type] || DEFAULT_SEGMENT_PROCESSING;

    return Object.entries(value[type]).some(
      ([key, val]) => val !== defaultConfig[key as keyof typeof defaultConfig]
    );
  };

  const updateConfig = (field: string, newValue: string) => {
    onChange({
      ...value,
      [selectedType]: {
        ...value[selectedType],
        [field]: newValue,
      },
    });
  };

  const handleTypeSelect = (type: keyof SegmentProcessing) => {
    setSelectedType(type);
    setIsDropdownOpen(false);
  };

  return (
    <div className="segment-processing-container">
      <div className="model-selector" ref={dropdownRef}>
        <button
          className="model-selector-button"
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
          type="button"
        >
          <Text size="2" weight="medium">
            {selectedType}
            {isSegmentModified(selectedType) && " (Modified)"}
          </Text>
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            style={{
              transform: isDropdownOpen ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.2s ease",
            }}
          >
            <path
              d="M2.5 4.5L6 8L9.5 4.5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
        {isDropdownOpen && (
          <div className="segment-dropdown-menu">
            {segmentTypes.map((type) => (
              <button
                key={type}
                className={`segment-dropdown-item ${
                  selectedType === type ? "active" : ""
                } ${isSegmentModified(type) ? "modified" : ""}`}
                onClick={() => handleTypeSelect(type)}
                type="button"
              >
                <Text size="2" weight="medium">
                  {type}
                  {isSegmentModified(type) && " (Modified)"}
                </Text>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="segment-config-grid">
        <ToggleGroup
          docHover={false}
          label="HTML Generation"
          value={value[selectedType].html}
          onChange={(v) => updateConfig("html", v)}
          options={[
            { label: "Auto", value: GenerationStrategy.Auto },
            { label: "LLM", value: GenerationStrategy.LLM },
          ]}
        />

        <ToggleGroup
          docHover={false}
          label="Markdown Generation"
          value={value[selectedType].markdown}
          onChange={(v) => updateConfig("markdown", v)}
          options={[
            { label: "Auto", value: GenerationStrategy.Auto },
            { label: "LLM", value: GenerationStrategy.LLM },
          ]}
        />

        <ToggleGroup
          docHover={false}
          label="Image Cropping"
          value={value[selectedType].crop_image}
          onChange={(v) => updateConfig("crop_image", v)}
          options={[
            { label: "Auto", value: CroppingStrategy.Auto },
            { label: "All", value: CroppingStrategy.All },
          ]}
        />
      </div>
    </div>
  );
}

interface JsonSchemaControlsProps {
  value: JsonSchema | undefined;
  onChange: (value: JsonSchema | undefined) => void;
}

export function JsonSchemaControls({
  value,
  onChange,
}: JsonSchemaControlsProps) {
  const [isPropertyOpen, setIsPropertyOpen] = useState<number | null>(null);

  const handleAddProperty = () => {
    const newSchema: JsonSchema = value ?? { title: "", properties: [] };
    onChange({
      ...newSchema,
      properties: [...newSchema.properties, { name: "", type: "string" }],
    });
  };

  const updateProperty = (index: number, updates: Partial<Property>) => {
    if (!value) return;
    const newProperties = [...value.properties];
    newProperties[index] = { ...newProperties[index], ...updates };
    onChange({ ...value, properties: newProperties });
  };

  const removeProperty = (index: number) => {
    if (!value) return;
    const newProperties = value.properties.filter((_, i) => i !== index);
    onChange({ ...value, properties: newProperties });
  };

  return (
    <div className="segment-processing-container">
      <div className="config-card">
        <div className="config-card-header">
          <Flex direction="row" gap="2" align="center">
            <svg
              width="24px"
              height="24px"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M14 19H16C17.1046 19 18 18.1046 18 17V14.5616C18 13.6438 18.6246 12.8439 19.5149 12.6213L21.0299 12.2425C21.2823 12.1794 21.2823 11.8206 21.0299 11.7575L19.5149 11.3787C18.6246 11.1561 18 10.3562 18 9.43845V5H14"
                stroke="#FFF"
                strokeWidth="2"
              />
              <path
                d="M10 5H8C6.89543 5 6 5.89543 6 7V9.43845C6 10.3562 5.37541 11.1561 4.48507 11.3787L2.97014 11.7575C2.71765 11.8206 2.71765 12.1794 2.97014 12.2425L4.48507 12.6213C5.37541 12.8439 6 13.6438 6 14.5616V19H10"
                stroke="#FFF"
                strokeWidth="2"
              />
            </svg>
            <Text size="3" weight="bold" className="white">
              JSON Schema
            </Text>
          </Flex>
          <Flex
            direction="row"
            gap="1"
            align="center"
            justify="end"
            className="docs-text"
          >
            <Text size="1" weight="medium" className="white ">
              Docs
            </Text>
            <svg
              width="12px"
              height="12px"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M14.1625 18.4876L13.4417 19.2084C11.053 21.5971 7.18019 21.5971 4.79151 19.2084C2.40283 16.8198 2.40283 12.9469 4.79151 10.5583L5.51236 9.8374"
                stroke="#FFFFFF"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              <path
                d="M9.8374 14.1625L14.1625 9.8374"
                stroke="#FFFFFF"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              <path
                d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
                stroke="#FFFFFF"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </Flex>
        </div>

        <input
          type="text"
          value={value?.title ?? ""}
          onChange={(e) =>
            onChange({
              ...value,
              title: e.target.value,
              properties: value?.properties ?? [],
            })
          }
          placeholder="Schema Title"
          className="number-input mb-4"
        />

        <div className="properties-list">
          {value?.properties.map((prop, index) => (
            <div key={index} className="property-item">
              <button
                className="model-selector-button mb-4"
                onClick={() =>
                  setIsPropertyOpen(isPropertyOpen === index ? null : index)
                }
                type="button"
              >
                <Text size="2" weight="medium">
                  {prop.name || "New Property"}
                </Text>
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  style={{
                    transform:
                      isPropertyOpen === index
                        ? "rotate(180deg)"
                        : "rotate(0deg)",
                    transition: "transform 0.2s ease",
                  }}
                >
                  <path
                    d="M2.5 4.5L6 8L9.5 4.5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>

              {isPropertyOpen === index && (
                <div className="property-details">
                  <input
                    type="text"
                    value={prop.name}
                    onChange={(e) =>
                      updateProperty(index, { name: e.target.value })
                    }
                    placeholder="Property Name"
                    className="number-input mb-4"
                  />
                  <input
                    type="text"
                    value={prop.title ?? ""}
                    onChange={(e) =>
                      updateProperty(index, { title: e.target.value })
                    }
                    placeholder="Property Title (Optional)"
                    className="number-input mb-4"
                  />
                  <select
                    value={prop.type}
                    onChange={(e) =>
                      updateProperty(index, { type: e.target.value })
                    }
                    className="number-input mb-4"
                  >
                    <option value="string">String</option>
                    <option value="number">Number</option>
                    <option value="boolean">Boolean</option>
                    <option value="array">Array</option>
                    <option value="object">Object</option>
                  </select>
                  <textarea
                    value={prop.description ?? ""}
                    onChange={(e) =>
                      updateProperty(index, { description: e.target.value })
                    }
                    placeholder="Description (Optional)"
                    className="number-input mb-4"
                    rows={3}
                  />
                  <input
                    type="text"
                    value={prop.default ?? ""}
                    onChange={(e) =>
                      updateProperty(index, { default: e.target.value })
                    }
                    placeholder="Default Value (Optional)"
                    className="number-input mb-4"
                  />
                  <button
                    onClick={() => removeProperty(index)}
                    className="delete-button"
                    type="button"
                  >
                    Remove Property
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        <button
          onClick={handleAddProperty}
          className="add-property-button"
          type="button"
        >
          Add Property
        </button>
      </div>
    </div>
  );
}

interface ChunkProcessingControlsProps {
  value: ChunkProcessing;
  onChange: (value: ChunkProcessing) => void;
}

export function ChunkProcessingControls({
  value,
  onChange,
}: ChunkProcessingControlsProps) {
  return (
    <div className="config-card">
      <div className="config-card-header">
        <Flex direction="row" gap="2" align="center">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <g clip-path="url(#clip0_305_31854)">
              <path
                d="M9.25 16C9.25 14.2051 7.79493 12.75 6 12.75C4.20507 12.75 2.75 14.2051 2.75 16C2.75 17.7949 4.20507 19.25 6 19.25C7.79493 19.25 9.25 17.7949 9.25 16Z"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M16.8699 4.75L8.85994 17.55L8.68994 17.82"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M14.75 16C14.75 17.7949 16.2051 19.25 18 19.25C19.7949 19.25 21.25 17.7949 21.25 16C21.25 14.2051 19.7949 12.75 18 12.75C16.2051 12.75 14.75 14.2051 14.75 16Z"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M15.3099 17.82L15.1399 17.55L7.12988 4.75"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
            <defs>
              <clipPath id="clip0_305_31854">
                <rect width="24" height="24" fill="white" />
              </clipPath>
            </defs>
          </svg>
          <Text size="3" weight="bold" className="white">
            Chunk Processing
          </Text>
        </Flex>

        <Flex
          direction="row"
          gap="1"
          align="center"
          justify="end"
          className="docs-text"
        >
          <Text size="1" weight="medium" className="white">
            Docs
          </Text>
          <svg
            width="12px"
            height="12px"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M14.1625 18.4876L13.4417 19.2084C11.053 21.5971 7.18019 21.5971 4.79151 19.2084C2.40283 16.8198 2.40283 12.9469 4.79151 10.5583L5.51236 9.8374"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M9.8374 14.1625L14.1625 9.8374"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </Flex>
      </div>

      <Flex direction="row" width="100%" justify="between" gap="24px">
        <Flex direction="row" width="fit-content" align="center" gap="2">
          <Text
            size="2"
            weight="medium"
            className="white"
            style={{ width: "fit-content" }}
          >
            Length:
          </Text>
          <input
            type="number"
            value={value.target_length}
            onChange={(e) =>
              onChange({ ...value, target_length: Number(e.target.value) })
            }
            min={0}
            className="number-input fit-content"
          />
        </Flex>

        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={value.ignore_headers_and_footers ?? false}
            onChange={(e) =>
              onChange({
                ...value,
                ignore_headers_and_footers: e.target.checked,
              })
            }
          />
          <Text size="1" weight="medium" className="white ml-2">
            Ignore Headers & Footers
          </Text>
        </label>
      </Flex>
    </div>
  );
}
