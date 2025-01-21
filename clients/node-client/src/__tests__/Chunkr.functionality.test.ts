import { Chunkr } from "../Chunkr";
import {
  Status,
  Configuration,
  SegmentationStrategy,
  GenerationStrategy,
  OcrStrategy,
  JsonSchema,
  Property,
} from "../models/Configuration";
import * as fs from "fs/promises";
import * as path from "path";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

const INPUT_DIR = "./input";
const OUTPUT_DIR = "./output";

// Helper function to get first file of specific type from input directory
async function getFirstFileOfType(extension: string): Promise<string> {
  const files = await fs.readdir(INPUT_DIR);
  const file = files.find((f) => f.toLowerCase().endsWith(extension));
  if (!file) {
    throw new Error(`No ${extension} file found in input directory`);
  }
  return path.join(INPUT_DIR, file);
}

describe("Chunkr Integration Tests", () => {
  let chunkr: Chunkr;
  let sampleFile: string;

  beforeAll(async () => {
    if (!process.env.CHUNKR_API_KEY) {
      throw new Error("CHUNKR_API_KEY not found in environment");
    }
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
    chunkr = new Chunkr(process.env.CHUNKR_API_KEY);
    // Get first PDF file from input directory
    sampleFile = await getFirstFileOfType(".pdf");
  });

  // Basic upload tests
  it("should upload file from path", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const response = await chunkr.createTask(fileContent);

    expect(response.task_id).toBeDefined();
    expect(response.status).toBe(Status.SUCCEEDED);
    expect(response.output).toBeDefined();
  });

  // Configuration tests
  it("should process with OCR auto strategy", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const config = new Configuration({ ocr_strategy: OcrStrategy.AUTO });
    const response = await chunkr.createTask(fileContent, config);

    expect(response.status).toBe(Status.SUCCEEDED);
    expect(response.output).toBeDefined();
  });

  it("should process with expiration time", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const config = new Configuration({ expires_in: 10 });
    const response = await chunkr.createTask(fileContent, config);

    expect(response.status).toBe(Status.SUCCEEDED);
    expect(response.output).toBeDefined();
  });

  it("should process with page segmentation strategy", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const config = new Configuration({
      segmentation_strategy: SegmentationStrategy.PAGE,
    });
    const response = await chunkr.createTask(fileContent, config);

    expect(response.status).toBe(Status.SUCCEEDED);
    expect(response.output).toBeDefined();
  });

  it("should process with LLM HTML generation", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const config = new Configuration({
      segmentation_strategy: SegmentationStrategy.PAGE,
      segment_processing: {
        page: {
          html: GenerationStrategy.LLM,
        },
      },
    });
    const response = await chunkr.createTask(fileContent, config);

    expect(response.status).toBe(Status.SUCCEEDED);
    expect(response.output).toBeDefined();
  });

  // Task management tests
  it("should delete a task", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const response = await chunkr.createTask(fileContent);
    expect(response.task_id).toBeDefined();

    await chunkr.deleteTask(response.task_id);

    // Verify task is deleted by checking it throws when fetching
    await expect(chunkr.getTask(response.task_id)).rejects.toThrow();
  });

  it("should cancel a task", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const response = await chunkr.createTask(fileContent);
    expect(response.task_id).toBeDefined();

    await chunkr.cancelTask(response.task_id);
    const updatedTask = await chunkr.getTask(response.task_id);
    expect(updatedTask.status).toBe(Status.CANCELLED);
  });

  it("should update task configuration", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const originalConfig = new Configuration({
      segmentation_strategy: SegmentationStrategy.LAYOUT_ANALYSIS,
    });

    const response = await chunkr.createTask(fileContent, originalConfig);
    expect(response.task_id).toBeDefined();

    const newConfig = new Configuration({
      segmentation_strategy: SegmentationStrategy.PAGE,
    });

    const updatedTask = await chunkr.updateTask(response.task_id, newConfig);
    expect(updatedTask.configuration.segmentation_strategy).toBe(
      SegmentationStrategy.PAGE,
    );
  });

  // JSON Schema tests
  it("should process with custom JSON schema", async () => {
    const fileContent = await fs.readFile(sampleFile);
    const config = new Configuration({
      json_schema: new JsonSchema({
        title: "Sales Data",
        properties: [
          new Property({
            name: "Person with highest sales",
            prop_type: "string",
            description: "The person with the highest sales",
          }),
          new Property({
            name: "Person with lowest sales",
            prop_type: "string",
            description: "The person with the lowest sales",
          }),
        ],
      }),
    });

    const response = await chunkr.createTask(fileContent, config);
    expect(response.status).toBe(Status.SUCCEEDED);
    expect(response.output).toBeDefined();
  }, 300000); // 5 minute timeout for longer runs
});
