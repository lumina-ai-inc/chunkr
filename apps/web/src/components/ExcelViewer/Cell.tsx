import { GridChildComponentProps } from "react-window";
import type { CellData } from "../../models/excelViewer.model";
import { Alignment, VerticalAlignment } from "../../models/taskResponse.model";
import type { CSSProperties } from "react";

const segmentTypeStyles: Record<
  string,
  { fill: string; dark: string; border: string }
> = {
  text: { fill: "jade-4", dark: "jade-11", border: "jade-10" },
  table: { fill: "orange-4", dark: "orange-11", border: "orange-9" },
  title: { fill: "blue-4", dark: "blue-11", border: "blue-9" },
  picture: { fill: "pink-4", dark: "pink-11", border: "pink-10" },
  formula: { fill: "amber-3", dark: "amber-11", border: "amber-8" },
  caption: { fill: "crimson-2", dark: "crimson-11", border: "crimson-8" },
  footnote: { fill: "pink-4", dark: "pink-11", border: "pink-10" },
  listitem: { fill: "bronze-4", dark: "bronze-11", border: "bronze-10" },
  pagefooter: { fill: "red-4", dark: "red-11", border: "red-12" },
  pageheader: { fill: "violet-4", dark: "violet-11", border: "violet-9" },
  sectionheader: { fill: "cyan-4", dark: "cyan-11", border: "cyan-8" },
  page: { fill: "gray-3", dark: "gray-11", border: "gray-8" },
};

const getColumnLetter = (index: number): string => {
  let letter = "";
  let temp = index;
  while (temp >= 0) {
    letter = String.fromCharCode((temp % 26) + 65) + letter;
    temp = Math.floor(temp / 26) - 1;
  }
  return letter;
};

function Cell({
  columnIndex,
  rowIndex,
  style,
  data,
}: GridChildComponentProps<CellData>) {
  const {
    rows,
    cells,
    highlights,
    formulas,
    formulaStrings,
    segmentRanges,
    hoveredHighlight,
    selectedHighlight,
    setHoveredHighlight,
    setHoveredFormula,
    onRangeClick,
    imageSegments,
    mergedRanges,
  } = data;

  if (rowIndex === 0 && columnIndex === 0) {
    return (
      <div
        style={{
          ...style,
          borderWidth: "0.5px",
        }}
        className="border-gray-200 bg-gray-50 flex items-center justify-center text-xs font-medium"
        onMouseEnter={() => {
          setHoveredHighlight(null);
          setHoveredFormula(null);
        }}
      ></div>
    );
  }

  if (rowIndex === 0) {
    return (
      <div
        style={{
          ...style,
          borderWidth: "0.5px",
        }}
        className="border-gray-200 bg-gray-50 flex items-center justify-center text-xs font-medium"
        onMouseEnter={() => {
          setHoveredHighlight(null);
          setHoveredFormula(null);
        }}
      >
        {getColumnLetter(columnIndex - 1)}
      </div>
    );
  }

  if (columnIndex === 0) {
    return (
      <div
        style={{
          ...style,
          borderWidth: "0.5px",
        }}
        className="border-gray-200 bg-gray-50 flex items-center justify-center text-xs font-medium"
        onMouseEnter={() => {
          setHoveredHighlight(null);
          setHoveredFormula(null);
        }}
      >
        {rowIndex}
      </div>
    );
  }

  const dataRowIndex = rowIndex - 1;
  const dataColIndex = columnIndex - 1;

  if (
    dataRowIndex >= rows.length ||
    dataColIndex >= (rows[dataRowIndex]?.length || 0)
  ) {
    return (
      <div
        style={{
          ...style,
        }}
        className="border-gray-200 bg-white border-[0.5px]"
        onMouseEnter={() => {
          setHoveredHighlight(null);
          setHoveredFormula(null);
        }}
      ></div>
    );
  }

  const cellValue = rows[dataRowIndex][dataColIndex];
  const cellData = cells[dataRowIndex]?.[dataColIndex];
  const highlightInfo = highlights[dataRowIndex]?.[dataColIndex];
  const isFormulaCell = formulas[dataRowIndex]?.[dataColIndex] || false;

  const imageSegment = imageSegments.find(
    (img) =>
      dataRowIndex >= img.range.startRow &&
      dataRowIndex <= img.range.endRow &&
      dataColIndex >= img.range.startCol &&
      dataColIndex <= img.range.endCol
  );

  let imageWidth = 0;
  let imageHeight = 0;
  let imageOffsetX = 0;
  let imageOffsetY = 0;

  if (imageSegment) {
    for (
      let c = imageSegment.range.startCol;
      c <= imageSegment.range.endCol;
      c++
    ) {
      imageWidth += data.columnWidths[c + 1] || 100;
    }

    const CELL_HEIGHT = 24;
    imageHeight =
      (imageSegment.range.endRow - imageSegment.range.startRow + 1) *
      CELL_HEIGHT;

    for (let c = imageSegment.range.startCol; c < dataColIndex; c++) {
      imageOffsetX -= data.columnWidths[c + 1] || 100;
    }

    imageOffsetY = -(dataRowIndex - imageSegment.range.startRow) * CELL_HEIGHT;
  }

  const handleMouseEnter = () => {
    if (highlightInfo) {
      const rangeInfo = segmentRanges.get(highlightInfo.id);
      if (rangeInfo) {
        const hr = rangeInfo.headerRange;
        const inHeader =
          hr &&
          dataRowIndex >= hr.startRow &&
          dataRowIndex <= hr.endRow &&
          dataColIndex >= hr.startCol &&
          dataColIndex <= hr.endCol;
        const cellType = inHeader
          ? `${highlightInfo.type}Header`
          : highlightInfo.type;
        const rng = inHeader ? hr! : rangeInfo.range;
        setHoveredHighlight({
          id: highlightInfo.id,
          type: cellType,
          range: rng,
        });
      }
    } else {
      setHoveredHighlight(null);
    }
    if (isFormulaCell) {
      setHoveredFormula(formulaStrings[dataRowIndex][dataColIndex] ?? null);
    } else {
      setHoveredFormula(null);
    }
  };

  const shouldShowTooltip =
    hoveredHighlight &&
    highlightInfo?.id === hoveredHighlight.id &&
    dataRowIndex === hoveredHighlight.range.startRow &&
    dataColIndex === hoveredHighlight.range.startCol;

  let borderTopColor = "#e5e7eb";
  let borderRightColor = "#e5e7eb";
  let borderBottomColor = "#e5e7eb";
  let borderLeftColor = "#e5e7eb";
  let borderTopWidth = "0.5px";
  let borderRightWidth = "0.5px";
  let borderBottomWidth = "0.5px";
  let borderLeftWidth = "0.5px";
  let backgroundColor = "white";

  // Apply cell-specific styles
  let cellBackgroundColor = backgroundColor;
  let cellTextColor = undefined;
  let cellFontFamily = undefined;
  let cellFontWeight = undefined;
  let cellTextAlign: Alignment | undefined = undefined;
  let cellVerticalAlign: VerticalAlignment | undefined = undefined;
  let isHyperlink = false;

  const mergeCellRange = mergedRanges.find(
    (range) =>
      dataRowIndex >= range.startRow &&
      dataRowIndex <= range.endRow &&
      dataColIndex >= range.startCol &&
      dataColIndex <= range.endCol
  );

  let mergedRangeBackgroundColor = undefined;
  if (mergeCellRange) {
    const primaryCellData =
      cells[mergeCellRange.startRow]?.[mergeCellRange.startCol];
    if (primaryCellData?.style?.bg_color) {
      mergedRangeBackgroundColor = primaryCellData.style.bg_color;
    }
  }

  if (cellData?.style) {
    if (cellData.style.bg_color && !mergeCellRange) {
      cellBackgroundColor = cellData.style.bg_color;
    }
    if (cellData.style.text_color) {
      cellTextColor = cellData.style.text_color;
    }
    if (cellData.style.font_face) {
      cellFontFamily = cellData.style.font_face;
    }
    if (cellData.style.is_bold) {
      cellFontWeight = "bold";
    }
    if (cellData.style.align) {
      cellTextAlign = cellData.style.align;
    }
    if (cellData.style.valign) {
      cellVerticalAlign = cellData.style.valign;
    }
  }

  if (mergedRangeBackgroundColor) {
    cellBackgroundColor = mergedRangeBackgroundColor;
  }

  if (cellData?.hyperlink) {
    isHyperlink = true;
  }

  if (highlightInfo) {
    const typeKey = highlightInfo.type.toLowerCase();
    const styleDef = segmentTypeStyles[typeKey];
    const rangeInfo = segmentRanges.get(highlightInfo.id);

    if (styleDef && rangeInfo) {
      const borderVar = `var(--${styleDef.border})`;
      const fullRange = rangeInfo.range;
      const headerRange = rangeInfo.headerRange;

      // Always apply colored borders for segments
      const applyRangeBorders = (range: typeof fullRange) => {
        if (dataRowIndex === range.startRow) {
          borderTopColor = borderVar;
          borderTopWidth = "2px";
        }
        if (dataRowIndex === range.endRow) {
          borderBottomColor = borderVar;
          borderBottomWidth = "2px";
        }
        if (dataColIndex === range.startCol) {
          borderLeftColor = borderVar;
          borderLeftWidth = "2px";
        }
        if (dataColIndex === range.endCol) {
          borderRightColor = borderVar;
          borderRightWidth = "2px";
        }
      };

      applyRangeBorders(fullRange);

      if (
        headerRange &&
        dataRowIndex >= headerRange.startRow &&
        dataRowIndex <= headerRange.endRow &&
        dataColIndex >= headerRange.startCol &&
        dataColIndex <= headerRange.endCol
      ) {
        const headerBorderVar = `var(--${styleDef.dark})`;
        const tempApplyHeaderBorders = (range: typeof fullRange) => {
          if (dataRowIndex === range.startRow) {
            borderTopColor = headerBorderVar;
            borderTopWidth = "2px";
          }
          if (dataRowIndex === range.endRow) {
            borderBottomColor = headerBorderVar;
            borderBottomWidth = "2px";
          }
          if (dataColIndex === range.startCol) {
            borderLeftColor = headerBorderVar;
            borderLeftWidth = "2px";
          }
          if (dataColIndex === range.endCol) {
            borderRightColor = headerBorderVar;
            borderRightWidth = "2px";
          }
        };
        tempApplyHeaderBorders(headerRange);
      }
    }
  }

  backgroundColor = cellBackgroundColor;

  if (mergeCellRange) {
    if (dataRowIndex > mergeCellRange.startRow) {
      borderTopWidth = "0px";
    }
    if (dataRowIndex < mergeCellRange.endRow) {
      borderBottomWidth = "0px";
    }
    if (dataColIndex > mergeCellRange.startCol) {
      borderLeftWidth = "0px";
    }
    if (dataColIndex < mergeCellRange.endCol) {
      borderRightWidth = "0px";
    }
  }

  const handleCellClick = () => {
    if (isHyperlink && cellData?.hyperlink) {
      window.open(cellData.hyperlink, "_blank", "noopener,noreferrer");
    } else if (highlightInfo) {
      onRangeClick?.(highlightInfo.id);
    }
  };

  // Map vertical alignment to flexbox alignment
  const getFlexAlignItems = (valign: VerticalAlignment | undefined) => {
    switch (valign) {
      case VerticalAlignment.Top:
        return "flex-start";
      case VerticalAlignment.Middle:
        return "center";
      case VerticalAlignment.Bottom:
        return "flex-end";
      case VerticalAlignment.Baseline:
        return "baseline";
      default:
        return "center"; // Default alignment
    }
  };

  return (
    <div
      style={{
        ...style,
        backgroundColor,
        borderTop: `${borderTopWidth} solid ${borderTopColor}`,
        borderRight: `${borderRightWidth} solid ${borderRightColor}`,
        borderBottom: `${borderBottomWidth} solid ${borderBottomColor}`,
        borderLeft: `${borderLeftWidth} solid ${borderLeftColor}`,
        textAlign: cellTextAlign as CSSProperties["textAlign"],
        alignItems: getFlexAlignItems(cellVerticalAlign),
      }}
      className={`flex px-2 text-xs text-left overflow-visible relative ${isHyperlink ? "cursor-pointer hover:underline" : ""
        }`}
      onMouseEnter={handleMouseEnter}
      onClick={handleCellClick}
      title={isHyperlink ? `${cellValue} (${cellData?.hyperlink})` : cellValue}
    >
      {highlightInfo &&
        (() => {
          const typeKey = highlightInfo.type.toLowerCase();
          const styleDef = segmentTypeStyles[typeKey];
          const isHovered =
            hoveredHighlight &&
            hoveredHighlight.id === highlightInfo.id &&
            dataRowIndex >= hoveredHighlight.range.startRow &&
            dataRowIndex <= hoveredHighlight.range.endRow &&
            dataColIndex >= hoveredHighlight.range.startCol &&
            dataColIndex <= hoveredHighlight.range.endCol;

          const isSelected = selectedHighlight?.id === highlightInfo.id;

          let isInSelectedSegmentHeader = false;
          if (selectedHighlight && !isSelected) {
            const selectedRangeInfo = segmentRanges.get(selectedHighlight.id);
            if (selectedRangeInfo?.headerRange) {
              const hr = selectedRangeInfo.headerRange;
              isInSelectedSegmentHeader =
                dataRowIndex >= hr.startRow &&
                dataRowIndex <= hr.endRow &&
                dataColIndex >= hr.startCol &&
                dataColIndex <= hr.endCol;
            }
          }

          const isActive = isHovered || isSelected || isInSelectedSegmentHeader;

          if (styleDef && isActive) {
            return (
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  backgroundColor: `color-mix(in srgb, var(--${styleDef.fill}) 20%, transparent)`,
                  zIndex: 0,
                }}
              />
            );
          }
          return null;
        })()}

      <div
        className="w-full truncate relative"
        style={{
          zIndex: 2,
          color: cellTextColor,
          fontFamily: cellFontFamily,
          fontWeight: cellFontWeight,
          paddingRight: isFormulaCell ? "4px" : undefined,
        }}
      >
        {cellValue}
      </div>
      {imageSegment && (
        <div
          className="absolute flex items-center justify-center p-1"
          style={{
            left: imageOffsetX,
            top: imageOffsetY,
            width: imageWidth,
            height: imageHeight,
            zIndex: 1,
          }}
        >
          <img
            src={imageSegment.imageUrl}
            alt={`Image segment ${imageSegment.id}`}
            className="max-w-full max-h-full object-contain"
          />
        </div>
      )}
      {isFormulaCell && (
        <span
          className="absolute bottom-0 right-0 w-3 h-3 flex items-center justify-center"
          style={{ marginRight: "1px", marginBottom: "1px" }}
        >
          <svg
            width="10"
            height="10"
            viewBox="0 0 90 90"
            xmlns="http://www.w3.org/2000/svg"
            className="fill-current text-gray-600"
          >
            <g transform="translate(0,90) scale(0.1,-0.1)">
              <path
                d="M445 800 c-34 -11 -90 -61 -110 -99 -12 -24 -23 -61 -39 -133 -6 -26
                -10 -28 -56 -28 -37 0 -50 -4 -50 -14 0 -8 -3 -21 -6 -30 -5 -13 3 -16 45 -16
                l52 0 -6 -37 c-4 -21 -17 -86 -30 -145 -35 -159 -60 -199 -85 -138 -24 58 -74
                76 -110 40 -47 -47 -4 -110 75 -110 146 0 218 88 256 313 l13 77 53 0 c39 0
                53 4 53 14 0 8 3 21 6 30 5 12 -5 16 -48 18 l-54 3 12 87 c7 48 19 99 27 114
                l14 26 20 -38 c25 -49 61 -64 97 -40 30 19 36 75 11 96 -20 16 -101 22 -140
                10z"
              />
              <path
                d="M583 405 c-38 -16 -45 -40 -10 -32 31 8 44 -2 62 -52 29 -82 14 -116
                  -44 -105 -55 11 -96 -53 -56 -86 24 -20 63 -3 91 40 29 44 44 51 44 20 0 -29
                  44 -70 75 -70 31 0 95 25 95 38 0 5 -13 7 -28 4 -35 -5 -45 8 -65 78 -15 50
                  -15 56 0 73 12 13 25 16 45 11 57 -12 104 52 63 86 -24 20 -63 3 -93 -40 -15
                  -22 -30 -39 -34 -40 -3 0 -9 13 -13 30 -12 55 -66 74 -132 45z"
              />
            </g>
          </svg>
        </span>
      )}
      {shouldShowTooltip &&
        (() => {
          const rawType = hoveredHighlight.type.toLowerCase();
          const typeKey = rawType.endsWith("header")
            ? rawType.slice(0, -6)
            : rawType;
          const finalTypeKey = segmentTypeStyles[typeKey] ? typeKey : rawType;

          return (
            <div
              className="absolute px-2 py-1 rounded text-xs shadow-lg pointer-events-none whitespace-nowrap z-10"
              style={{
                color: `var(--${segmentTypeStyles[finalTypeKey]?.dark || "gray-11"
                  })`,
                top: "-100%",
                left: "0",
                transform: "translateY(-4px)",
                backgroundColor: `color-mix(in srgb, var(--${segmentTypeStyles[finalTypeKey]?.fill || "gray-3"
                  }) 100%, transparent)`,
                border: `1px solid var(--${segmentTypeStyles[finalTypeKey]?.border || "gray-8"
                  })`,
              }}
            >
              {hoveredHighlight.type}
            </div>
          );
        })()}
    </div>
  );
}

export default Cell;
