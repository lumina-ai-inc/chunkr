import Stack from "@mui/material/Stack";
import { Gauge } from "@mui/x-charts/Gauge";

export default function GaugeValueRangeNoSnap() {
  return (
    <Stack direction={{ xs: "column", md: "row" }} spacing={{ xs: 1, md: 3 }}>
      <Gauge
        width={164}
        height={164}
        value={75}
        valueMin={0}
        valueMax={1000}
        startAngle={0}
        endAngle={360}
        innerRadius="80%"
        outerRadius="100%"
        sx={{
          fontSize: 20,
          fontWeight: "medium",
        }}
        text={({ value, valueMax }) => `${value} / ${valueMax}`}
      />
    </Stack>
  );
}
