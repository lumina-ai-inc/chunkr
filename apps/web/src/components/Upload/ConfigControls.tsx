import { Flex, Text } from "@radix-ui/themes";
import "./ConfigControls.css";
import { useState, useEffect, useRef } from "react";
import {
  SegmentProcessing,
  GenerationStrategy,
  CroppingStrategy,
  DEFAULT_SEGMENT_PROCESSING,
  ChunkProcessing,
  LlmProcessing,
  FallbackStrategyType,
  EmbedSource,
  Tokenizer,
} from "../../models/taskConfig.model";
import { fetchLLMModels, LLMModel } from "../../services/llmModels.service";

interface ToggleGroupProps {
  value: string;
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  label: React.ReactNode;
  docsUrl?: string;
  docHover?: boolean;
}

export function ToggleGroup({
  value,
  onChange,
  options,
  label,
  docHover = true,
  docsUrl,
}: ToggleGroupProps) {
  return (
    <div className="config-card">
      <div className="config-card-header">
        <Text size="3" weight="bold" className="white">
          {label}
        </Text>
        <Flex
          onClick={() => docsUrl && window.open(docsUrl, "_blank")}
          direction="row"
          gap="1"
          align="center"
          justify="end"
          className={docHover ? "docs-text" : "docs-text-hidden"}
        >
          <Text size="1" weight="bold" className="white ">
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
            className={`toggle-button ${
              value === option.value ? "active" : ""
            }`}
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
          <Text size="1" weight="bold" className="white ">
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
  const segmentTypes = showOnlyPage
    ? (["Page"] as (keyof SegmentProcessing)[])
    : (Object.keys(value)
        .filter((key) => key !== "Page")
        .sort() as (keyof SegmentProcessing)[]);

  const defaultSegmentType = segmentTypes[0];
  const [selectedType, setSelectedType] =
    useState<keyof SegmentProcessing>(defaultSegmentType);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (showOnlyPage && selectedType !== "Page") {
      setSelectedType("Page");
    } else if (!showOnlyPage && selectedType === "Page") {
      setSelectedType(defaultSegmentType);
    }
  }, [selectedType, showOnlyPage, defaultSegmentType]);

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

  const isModified = (type: keyof SegmentProcessing) => {
    const defaultConfig = DEFAULT_SEGMENT_PROCESSING[type];
    const currentConfig = value[type];

    const defaultSources = defaultConfig.embed_sources ?? [
      EmbedSource.MARKDOWN,
    ];
    const currentSources = currentConfig.embed_sources ?? [
      EmbedSource.MARKDOWN,
    ];

    return (
      defaultConfig.crop_image !== currentConfig.crop_image ||
      defaultConfig.html !== currentConfig.html ||
      defaultConfig.markdown !== currentConfig.markdown ||
      (!!currentConfig.llm && currentConfig.llm.trim() !== "") ||
      JSON.stringify(defaultSources) !== JSON.stringify(currentSources)
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

  // derive the current embed_sources for this segment type:
  const currentSources: EmbedSource[] = value[selectedType].embed_sources ?? [
    EmbedSource.MARKDOWN,
  ];

  // helper to add/remove a source
  const toggleEmbedSource = (src: EmbedSource) => {
    const has = currentSources.includes(src);
    const newSources = has
      ? currentSources.filter((s) => s !== src)
      : [...currentSources, src];

    onChange({
      ...value,
      [selectedType]: {
        ...value[selectedType],
        embed_sources: newSources,
      },
    });
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
            {isModified(selectedType) && " (Modified)"}
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
                } ${isModified(type) ? "modified" : ""}`}
                onClick={() => handleTypeSelect(type)}
                type="button"
              >
                <Text size="2" weight="medium">
                  {type}
                  {isModified(type) && " (Modified)"}
                </Text>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="segment-config-grid">
        {/** ==== Markdown Generation ==== **/}
        <div className="config-card">
          <div
            className="config-card-header"
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text size="3" weight="bold" className="white">
              Markdown Generation
            </Text>

            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={currentSources.includes(EmbedSource.MARKDOWN)}
                onChange={() => toggleEmbedSource(EmbedSource.MARKDOWN)}
              />
            </label>
          </div>

          <div className="toggle-buttons">
            {[
              { label: "Auto", value: GenerationStrategy.Auto },
              { label: "LLM", value: GenerationStrategy.LLM },
            ].map((option) => (
              <button
                key={option.value}
                className={`toggle-button ${
                  value[selectedType].markdown === option.value ? "active" : ""
                }`}
                onClick={() =>
                  onChange({
                    ...value,
                    [selectedType]: {
                      ...value[selectedType],
                      markdown: option.value,
                    },
                  })
                }
              >
                <Text size="1" weight="bold">
                  {option.label}
                </Text>
              </button>
            ))}
          </div>
        </div>

        {/** ==== HTML Generation ==== **/}
        <div className="config-card">
          <div
            className="config-card-header"
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text size="3" weight="bold" className="white">
              HTML Generation
            </Text>

            {/** embed_sources checkbox **/}
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={currentSources.includes(EmbedSource.HTML)}
                onChange={() => toggleEmbedSource(EmbedSource.HTML)}
              />
            </label>
          </div>

          <div className="toggle-buttons">
            {[
              { label: "Auto", value: GenerationStrategy.Auto },
              { label: "LLM", value: GenerationStrategy.LLM },
            ].map((option) => (
              <button
                key={option.value}
                className={`toggle-button ${
                  value[selectedType].html === option.value ? "active" : ""
                }`}
                onClick={() =>
                  onChange({
                    ...value,
                    [selectedType]: {
                      ...value[selectedType],
                      html: option.value,
                    },
                  })
                }
              >
                <Text size="1" weight="bold">
                  {option.label}
                </Text>
              </button>
            ))}
          </div>
        </div>

        {/** ==== Custom LLM ==== **/}
        <div className="config-card">
          <div
            className="config-card-header"
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text size="3" weight="bold" className="white">
              Custom LLM
            </Text>

            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={currentSources.includes(EmbedSource.LLM)}
                onChange={() => toggleEmbedSource(EmbedSource.LLM)}
              />
            </label>
          </div>

          <input
            type="text"
            className="llm-input"
            placeholder="Enter custom prompt..."
            value={value[selectedType].llm || ""}
            onChange={(e) =>
              onChange({
                ...value,
                [selectedType]: {
                  ...value[selectedType],
                  llm: e.target.value,
                },
              })
            }
          />
        </div>

        {/** ==== Image Cropping (unchanged) ==== **/}
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

export function ChunkProcessingControls({
  value,
  onChange,
  docsUrl,
}: {
  value: ChunkProcessing;
  onChange: (value: ChunkProcessing) => void;
  docsUrl?: string;
}) {
  // Dropdown state for Tokenizer picker
  const [isTokOpen, setIsTokOpen] = useState(false);
  const tokRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (tokRef.current && !tokRef.current.contains(e.target as Node)) {
        setIsTokOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Predefined tokens per docs
  const predefined: { label: string; value: Tokenizer }[] = [
    { label: "Word", value: Tokenizer.Word },
    { label: "CL100K Base", value: Tokenizer.CL100K_BASE },
    { label: "XLM‑Roberta Base", value: Tokenizer.XLM_ROBERTA_BASE },
    { label: "BERT Base Uncased", value: Tokenizer.BERT_BASE_UNCASED },
  ];

  // Determine the current tokenizer value and if it's custom
  const currentTokenizerValue = value.tokenizer?.Enum ?? Tokenizer.Word; // Get the value inside Enum, default to Word
  const isCustom = !predefined.find((p) => p.value === currentTokenizerValue);

  // Handler for selecting a tokenizer
  const selectTokenizer = (tok: Tokenizer | string) => {
    onChange({ ...value, tokenizer: { Enum: tok } }); // Ensure the object structure is always set
    setIsTokOpen(false);
  };

  return (
    <div
      className="chunk-processing-container config-card"
      style={{ zIndex: 100, position: "relative", marginTop: "40px" }}
    >
      {/* === Parent Header === */}
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
        {docsUrl && (
          <Flex
            onClick={() => window.open(docsUrl, "_blank")}
            direction="row"
            gap="1"
            align="center"
            justify="end"
            className="docs-text"
          >
            <Text size="1" weight="bold" className="white">
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
        )}
      </div>

      {/* === Sub‑cards grid === */}
      <div className="segment-config-grid">
        {/* 1) Length sub–card */}
        <div className="config-card">
          <div className="config-card-header">
            <Text size="3" weight="bold" className="white">
              Length
            </Text>
          </div>
          <input
            type="number"
            value={value.target_length}
            min={0}
            onChange={(e) =>
              onChange({ ...value, target_length: +e.target.value })
            }
            className="number-input"
          />
        </div>

        {/* 2) Tokenizer sub–card */}
        <div className="config-card">
          <div className="config-card-header">
            <Text size="3" weight="bold" className="white">
              Tokenizer
            </Text>
          </div>

          {/* Re‑use the LLM "model selector" dropdown pattern */}
          <div className="model-selector" ref={tokRef}>
            <button
              className="model-selector-button"
              type="button"
              onClick={() => setIsTokOpen((o) => !o)}
            >
              <Text size="2" weight="medium">
                {/* Display based on the actual value */}
                {isCustom ? "Custom…" : currentTokenizerValue}
              </Text>
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                style={{
                  transform: isTokOpen ? "rotate(180deg)" : "rotate(0deg)",
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
            {isTokOpen && (
              <div className="segment-dropdown-menu" style={{ zIndex: 100 }}>
                {predefined.map((opt) => (
                  <button
                    key={opt.value}
                    className={`segment-dropdown-item ${
                      // Compare with the actual value
                      currentTokenizerValue === opt.value ? "active" : ""
                    }`}
                    onClick={() => selectTokenizer(opt.value)}
                  >
                    <Text size="2" weight="medium">
                      {opt.label}
                      {opt.value === Tokenizer.Word && " (Default)"}
                    </Text>
                  </button>
                ))}
                <button
                  key="__custom__"
                  className={`segment-dropdown-item ${
                    isCustom ? "active" : ""
                  }`}
                  onClick={() => selectTokenizer("")}
                >
                  <Text size="2" weight="medium">
                    Custom
                  </Text>
                </button>
              </div>
            )}
          </div>

          {/* If custom, show free‑form input */}
          {isCustom && (
            <input
              type="text"
              placeholder="huggingface/model‑id"
              // Bind to the actual value
              value={currentTokenizerValue as string}
              onChange={(e) => selectTokenizer(e.target.value)}
              className="number-input"
              style={{ marginTop: "8px" }}
            />
          )}
        </div>

        {/* 3) Ignore Headers & Footers sub–card */}
        <ToggleGroup
          label="Headers & Footers"
          value={String(!!value.ignore_headers_and_footers)}
          onChange={(newValue) =>
            onChange({
              ...value,
              ignore_headers_and_footers: newValue === "true",
            })
          }
          options={[
            { label: "Ignore", value: "true" },
            { label: "Include", value: "false" },
          ]}
          docHover={false}
        />
      </div>
    </div>
  );
}

export function LlmProcessingControls({
  value,
  onChange,
  docsUrl,
}: {
  value: LlmProcessing;
  onChange: (value: LlmProcessing) => void;
  docsUrl?: string;
}) {
  const [models, setModels] = useState<LLMModel[]>([]);
  const [loading, setLoading] = useState(true);

  // Dropdown state
  const [isModelOpen, setIsModelOpen] = useState(false);
  const [isFallbackOpen, setIsFallbackOpen] = useState(false);
  const modelRef = useRef<HTMLDivElement>(null);
  const fallbackRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    fetchLLMModels()
      .then(setModels)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // close dropdowns on outside click
  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (modelRef.current && !modelRef.current.contains(e.target as Node)) {
        setIsModelOpen(false);
      }
      if (
        fallbackRef.current &&
        !fallbackRef.current.contains(e.target as Node)
      ) {
        setIsFallbackOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  if (loading) {
    return (
      <div className="config-card">
        <Text size="2" weight="medium" className="white">
          Loading LLM models…
        </Text>
      </div>
    );
  }

  // figure out the system defaults
  const defaultModelId = models.find((m) => m.default)?.id || "";
  const defaultFallbackId = models.find((m) => m.fallback)?.id || "";

  // determine what's currently shown
  const selectedModelId = value.model_id ?? defaultModelId;

  // Show "Default" in the button label when the default model is selected
  const selectedModelDisplayText =
    selectedModelId + (selectedModelId === defaultModelId ? " (Default)" : "");

  // --- Updated Fallback Logic ---
  // Get the type (key) and model ID (value) from the fallback_strategy object
  const fallbackStrategy = value.fallback_strategy ?? {
    [FallbackStrategyType.Default]: null,
  };
  const currentFallbackType = Object.keys(
    fallbackStrategy
  )[0] as FallbackStrategyType;
  // Conditionally access the model ID only if the type is 'Model'
  const currentFallbackId =
    currentFallbackType === FallbackStrategyType.Model
      ? (fallbackStrategy as { [FallbackStrategyType.Model]: string })[
          FallbackStrategyType.Model
        ]
      : "";

  // Determine the text to display for the fallback button
  let fallbackDisplayText = "";
  if (currentFallbackType === FallbackStrategyType.Default) {
    fallbackDisplayText = `${defaultFallbackId} (Default)`;
  } else if (currentFallbackType === FallbackStrategyType.Model) {
    fallbackDisplayText = currentFallbackId;
  } else {
    // FallbackStrategyType.None or unexpected
    fallbackDisplayText = "None"; // Or handle other cases as needed
  }

  return (
    <div className="config-card">
      {/* Top Header */}
      <div className="config-card-header">
        <Flex direction="row" gap="2" align="center">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M19 3V7M17 5H21M19 17V21M17 19H21M10 5L8.53001 8.72721C8.3421 9.20367 8.24814 9.4419 8.10427 9.64278C7.97675 9.82084 7.82084 9.97675 7.64278 10.1043C7.4419 10.2481 7.20367 10.3421 6.72721 10.53L3 12L6.72721 13.47C7.20367 13.6579 7.4419 13.7519 7.64278 13.8957C7.82084 14.0233 7.97675 14.1792 8.10427 14.3572C8.24814 14.5581 8.3421 14.7963 8.53001 15.2728L10 19L11.47 15.2728C11.6579 14.7963 11.7519 14.5581 11.8957 14.3572C12.0233 14.1792 12.1792 14.0233 12.3572 13.8957C12.5581 13.7519 12.7963 13.6579 13.2728 13.47L17 12L13.2728 10.53C12.7963 10.3421 12.5581 10.2481 12.3572 10.1043C12.1792 9.97675 12.0233 9.82084 11.8957 9.64278C11.7519 9.4419 11.6579 9.20367 11.47 8.72721L10 5Z"
              stroke="#FFF"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <Text size="3" weight="bold" className="white">
            LLM Processing
          </Text>
        </Flex>
        {docsUrl && (
          <Flex
            onClick={() => window.open(docsUrl, "_blank")}
            direction="row"
            gap="1"
            align="center"
            justify="end"
            className="docs-text"
          >
            <Text size="1" weight="bold" className="white">
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
        )}
      </div>

      <div className="segment-config-grid">
        {/* ==== MAIN MODEL SELECTOR ==== */}
        <div
          className="config-card"
          style={{
            zIndex: 100,
          }}
        >
          <div className="config-card-header">
            <Text size="3" weight="bold" className="white">
              Model
            </Text>
          </div>
          <div className="model-selector" ref={modelRef}>
            <button
              className="model-selector-button"
              onClick={() => setIsModelOpen((o) => !o)}
              type="button"
            >
              <Text size="2" weight="medium">
                {selectedModelDisplayText}
              </Text>
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                style={{
                  transform: isModelOpen ? "rotate(180deg)" : "rotate(0deg)",
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
            {isModelOpen && (
              <div className="segment-dropdown-menu">
                {models.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    className={`segment-dropdown-item ${
                      selectedModelId === m.id ? "active" : ""
                    }`}
                    onClick={() => {
                      onChange({
                        ...value,
                        // if it's the system default, clear it, else pick it
                        model_id: m.id === defaultModelId ? undefined : m.id,
                      });
                      setIsModelOpen(false);
                    }}
                  >
                    <Text size="2" weight="medium">
                      {m.id}
                      {m.default && " (Default)"}
                    </Text>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ==== FALLBACK SELECTOR ==== */}
        <div
          className="config-card"
          style={{
            zIndex: 100,
            position: "relative",
          }}
        >
          <div className="config-card-header">
            <Text size="3" weight="bold" className="white">
              Fallback
            </Text>
          </div>
          <div className="model-selector" ref={fallbackRef}>
            <button
              className="model-selector-button"
              onClick={() => setIsFallbackOpen((o) => !o)}
              type="button"
            >
              <Text size="2" weight="medium">
                {fallbackDisplayText}
              </Text>
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                style={{
                  transform: isFallbackOpen ? "rotate(180deg)" : "rotate(0deg)",
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

            {isFallbackOpen && (
              <div className="segment-dropdown-menu">
                {/** 1) Always show the system default fallback **/}
                <button
                  type="button"
                  className={`segment-dropdown-item ${
                    currentFallbackType === FallbackStrategyType.Default
                      ? "active"
                      : ""
                  }`}
                  onClick={() => {
                    onChange({
                      ...value,
                      fallback_strategy: {
                        [FallbackStrategyType.Default]: null,
                      },
                    });
                    setIsFallbackOpen(false);
                  }}
                >
                  <Text size="2" weight="medium">
                    {defaultFallbackId}
                    {" (Default)"}
                  </Text>
                </button>

                {/** 2) Then any other model (excluding the default fallback) **/}
                {models
                  .filter((m) => m.id !== defaultFallbackId)
                  .map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      className={`segment-dropdown-item ${
                        currentFallbackType === FallbackStrategyType.Model &&
                        currentFallbackId === m.id
                          ? "active"
                          : ""
                      }`}
                      onClick={() => {
                        onChange({
                          ...value,
                          fallback_strategy: {
                            [FallbackStrategyType.Model]: m.id,
                          },
                        });
                        setIsFallbackOpen(false);
                      }}
                    >
                      <Text size="2" weight="medium">
                        {m.id}
                      </Text>
                    </button>
                  ))}
              </div>
            )}
          </div>
        </div>

        {/* ==== Temperature ==== */}
        <div className="config-card">
          <div className="config-card-header">
            <Text size="3" weight="bold" className="white">
              Temperature
            </Text>
          </div>
          <input
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={value.temperature ?? 0}
            onChange={(e) =>
              onChange({ ...value, temperature: parseFloat(e.target.value) })
            }
            className="number-input"
          />
        </div>

        {/* ==== Max Completion Tokens ==== */}
        <div className="config-card">
          <div className="config-card-header">
            <Text size="3" weight="bold" className="white">
              Max Completion Tokens
            </Text>
          </div>
          <input
            type="number"
            value={value.max_completion_tokens ?? ""}
            onChange={(e) => {
              const str = e.target.value;
              onChange({
                ...value,
                max_completion_tokens: str === "" ? undefined : Number(str),
              });
            }}
            min={0}
            className="number-input"
          />
        </div>
      </div>
    </div>
  );
}
