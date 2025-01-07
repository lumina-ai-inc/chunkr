import { Flex, Text } from "@radix-ui/themes";
import "./ConfigControls.css";
import { useState, useEffect, useRef } from "react";
import {
  SegmentProcessing,
  GenerationStrategy,
  CroppingStrategy,
  Property,
  JsonSchema,
} from "../../models/newTask.model";

interface ToggleGroupProps {
  value: string;
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  label: string;
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
              stroke-width="1.5"
              stroke-linecap="round"
            />
            <path
              d="M9.8374 14.1625L14.1625 9.8374"
              stroke="#FFFFFF"
              stroke-width="1.5"
              stroke-linecap="round"
            />
            <path
              d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
              stroke="#FFFFFF"
              stroke-width="1.5"
              stroke-linecap="round"
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
  label: string;
  min?: number;
  max?: number;
}

export function NumberInput({
  value,
  onChange,
  label,
  min,
  max,
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
              stroke-width="1.5"
              stroke-linecap="round"
            />
            <path
              d="M9.8374 14.1625L14.1625 9.8374"
              stroke="#FFFFFF"
              stroke-width="1.5"
              stroke-linecap="round"
            />
            <path
              d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
              stroke="#FFFFFF"
              stroke-width="1.5"
              stroke-linecap="round"
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
}

export function SegmentProcessingControls({
  value,
  onChange,
}: SegmentProcessingControlsProps) {
  const [selectedType, setSelectedType] =
    useState<keyof SegmentProcessing>("Text");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const segmentTypes = Object.keys(value) as (keyof SegmentProcessing)[];
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  // Check if segment has non-default settings
  const isSegmentModified = (type: keyof SegmentProcessing) => {
    const defaultConfig = {
      html: GenerationStrategy.Auto,
      markdown: GenerationStrategy.Auto,
      crop_image: CroppingStrategy.Auto,
    };

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
          <Text size="3" weight="bold" className="white">
            JSON Schema
          </Text>
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
                stroke-width="1.5"
                stroke-linecap="round"
              />
              <path
                d="M9.8374 14.1625L14.1625 9.8374"
                stroke="#FFFFFF"
                stroke-width="1.5"
                stroke-linecap="round"
              />
              <path
                d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
                stroke="#FFFFFF"
                stroke-width="1.5"
                stroke-linecap="round"
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
