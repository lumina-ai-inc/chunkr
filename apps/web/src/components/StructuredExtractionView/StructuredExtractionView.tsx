import { TaskResponse } from "../../models/taskResponse.model";
import { Text, Flex } from "@radix-ui/themes";
import "./StructuredExtractionView.css";

export default function StructuredExtractionView({
  task,
}: {
  task: TaskResponse;
}) {
  const extractedData = task.output?.extracted_json;

  if (!extractedData) {
    return null;
  }

  return (
    <Flex p="16px" width="100%" height="100%">
      <div className="extraction-container">
        <Text
          size="6"
          weight="bold"
          ml="4px"
          style={{ color: "rgba(255, 255, 255, 0.95)" }}
        >
          {extractedData.title}
        </Text>
        <div className="extraction-fields">
          {extractedData.extracted_fields.map((field) => (
            <div key={field.name} className="field-item">
              <Text size="2" className="field-label">
                {field.name}
              </Text>
              <Text size="2" className="field-type">
                {field.field_type}
              </Text>
              <Text size="3" mt="2" className="field-value">
                {String(field.value)}
              </Text>
            </div>
          ))}
        </div>
      </div>
    </Flex>
  );
}
