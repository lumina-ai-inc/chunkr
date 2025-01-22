import { Chunkr } from "../Chunkr";
import { Status } from "../models/TaskResponseData";
import { describe, it, expect, beforeAll } from "@jest/globals";
import * as path from "path";
import { SegmentationStrategy, OcrStrategy } from "../models/Configuration";

describe("Chunkr Basic Functionality", () => {
  let chunkr: Chunkr;
  let uploadedTaskId: string;
  const TEST_FILES_DIR = path.join(__dirname, "input");

  beforeAll(() => {
    chunkr = new Chunkr();
    if (!process.env.CHUNKR_API_KEY) {
      throw new Error("CHUNKR_API_KEY not found in environment");
    }

    // Check if input directory exists
    if (!require("fs").existsSync(TEST_FILES_DIR)) {
      throw new Error(`Input directory not found at: ${TEST_FILES_DIR}`);
    }

    // Check if directory contains any files
    const files = require("fs").readdirSync(TEST_FILES_DIR);
    if (files.length === 0) {
      throw new Error(`No files found in input directory: ${TEST_FILES_DIR}`);
    }
  });

  it("should successfully upload and process a single file", async () => {
    // Get all files from the input directory
    const files = require("fs").readdirSync(TEST_FILES_DIR);
    if (files.length === 0) {
      throw new Error("No files found in input directory");
    }

    // Pick a random file from the directory
    const randomFile = files[Math.floor(Math.random() * files.length)];
    const testFilePath = path.join(TEST_FILES_DIR, randomFile);

    try {
      // Upload the file and wait for processing to complete
      const result = await chunkr.upload(testFilePath);

      // Store the task ID for the next test
      uploadedTaskId = result.task_id;

      if (result.status === Status.SUCCEEDED) {
        console.log("File processed successfully:", {
          taskId: result.task_id,
          fileName: result.file_name,
          status: result.status,
          pageCount: result.page_count,
        });
      } else {
        console.error("File processing failed:", {
          taskId: result.task_id,
          fileName: result.file_name,
          status: result.status,
          error: result.error,
        });
        expect(result.status).toBe(Status.SUCCEEDED);
      }

      expect(result.status).toBe(Status.SUCCEEDED);
      expect(result.page_count).toBeGreaterThan(0);
      expect(result.output?.chunks.length).toBeGreaterThan(0);
    } catch (error) {
      console.error("Test failed:", error);
      throw error;
    }
  });

  it("should successfully update a file", async () => {
    // Get the existing task
    const task = await chunkr.getTask(uploadedTaskId);

    // Update the task with new configuration
    const updateResult = await chunkr.updateTask(uploadedTaskId, {
      high_resolution: true,
    });

    // Wait for the update to complete
    const finalResult = await updateResult.poll();

    // Verify the update was successful
    expect(finalResult.status).toBe(Status.SUCCEEDED);
    expect(finalResult.task_id).toBe(uploadedTaskId);
    expect(finalResult.output?.chunks.length).toBeGreaterThan(0);
  });

  it("should successfully delete a file", async () => {
    // Delete the task
    await chunkr.deleteTask(uploadedTaskId);

    // Try to get the task and expect it to fail
    try {
      await chunkr.getTask(uploadedTaskId);
      expect(false).toBe(true);
    } catch (error: any) {
      // Verify the error indicates the task was not found
      expect(error.response?.status).toBe(404);
    }
  });
});

describe("Chunkr Advanced Functionality", () => {
  let chunkr: Chunkr;
  const TEST_FILES_DIR = path.join(__dirname, "input");

  beforeAll(() => {
    chunkr = new Chunkr();
  });

  it("should process files with different segmentation strategies", async () => {
    // Get a test file
    const files = require("fs").readdirSync(TEST_FILES_DIR);
    const testFilePath = path.join(TEST_FILES_DIR, files[0]);

    const strategies = [
      SegmentationStrategy.LAYOUT_ANALYSIS,
      SegmentationStrategy.PAGE,
    ];

    console.log("\nSegmentation Strategy Test Results:");
    console.log("=================================");

    for (const strategy of strategies) {
      try {
        const result = await chunkr.upload(testFilePath, {
          segmentation_strategy: strategy,
        });

        // Create a formatted test result output
        const testResult = {
          Strategy: strategy,
          "Task ID": result.task_id,
          "File Name": result.file_name,
          Status: result.status,
          "Page Count": result.page_count,
          "Chunk Count": result.output?.chunks.length,
        };

        // Log the results in a structured format
        console.log(`\n${strategy} Strategy Results:`);
        console.table(testResult);

        // Basic validations
        expect(result.status).toBe(Status.SUCCEEDED);
        expect(result.output?.chunks.length).toBeGreaterThan(0);

        // Clean up
        await chunkr.deleteTask(result.task_id);
      } catch (error) {
        console.error(`Failed for strategy: ${strategy}`, error);
        throw error;
      }
    }
  });

  it("should process files with different OCR strategies", async () => {
    // Get a test file
    const files = require("fs").readdirSync(TEST_FILES_DIR);
    const testFilePath = path.join(TEST_FILES_DIR, files[0]);

    const strategies = [OcrStrategy.AUTO, OcrStrategy.ALL];

    console.log("\nOCR Strategy Test Results:");
    console.log("==========================");

    for (const strategy of strategies) {
      try {
        const result = await chunkr.upload(testFilePath, {
          ocr_strategy: strategy,
        });

        // Count segments with OCR results
        const segmentsWithOcr =
          result.output?.chunks.reduce((count, chunk) => {
            return (
              count +
              chunk.segments.filter(
                (segment) => segment.ocr && segment.ocr.length > 0,
              ).length
            );
          }, 0) || 0;

        // Check if OCR results have valid bounding boxes
        const hasValidBoundingBoxes = result.output?.chunks.every((chunk) =>
          chunk.segments.every((segment) =>
            segment.ocr?.every(
              (ocr) =>
                ocr.bbox &&
                typeof ocr.bbox.left === "number" &&
                typeof ocr.bbox.top === "number" &&
                typeof ocr.bbox.width === "number" &&
                typeof ocr.bbox.height === "number",
            ),
          ),
        );

        // Create a formatted test result output
        const testResult = {
          Strategy: strategy,
          "Task ID": result.task_id,
          Status: result.status,
          "Segments with OCR": segmentsWithOcr,
          "Valid Bounding Boxes": hasValidBoundingBoxes ? "Yes" : "No",
          "OCR Confidence":
            (result.output?.chunks
              ?.flatMap((c) => c.segments)
              ?.flatMap((s) => s.ocr || [])
              ?.filter((o) => o.confidence !== null)
              ?.map((o) => o.confidence)?.length ?? 0) > 0
              ? "Present"
              : "None",
        };

        // Log the results in a structured format
        console.log(`\n${strategy} Strategy Results:`);
        console.table(testResult);

        // Basic validations
        expect(result.status).toBe(Status.SUCCEEDED);

        // Strategy-specific validations
        if (strategy === OcrStrategy.ALL) {
          // ALL strategy should have OCR results for segments
          expect(segmentsWithOcr).toBeGreaterThan(0);
          expect(hasValidBoundingBoxes).toBe(true);
        }

        // Clean up
        await chunkr.deleteTask(result.task_id);
      } catch (error) {
        console.error(`Failed for OCR strategy: ${strategy}`, error);
        throw error;
      }
    }
  });
});
