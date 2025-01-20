import { Chunkr } from "../Chunkr";
import axios from "axios";
import { Status } from "../models/Configuration";

// Mock axios
jest.mock("axios");
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe("Chunkr Client", () => {
  let chunkr: Chunkr;

  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
    chunkr = new Chunkr("test-api-key");
  });

  describe("initialization", () => {
    it("should create instance with API key", () => {
      expect(chunkr).toBeInstanceOf(Chunkr);
    });

    it("should throw error without API key", () => {
      expect(() => new Chunkr()).toThrow();
    });
  });

  describe("upload", () => {
    it("should successfully upload a file", async () => {
      // Mock successful response
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          task_id: "123",
          status: Status.SUCCEEDED,
          result: { text: "processed content" },
        },
      });

      const result = await chunkr.upload(Buffer.from("test content"));

      expect(result.taskId).toBe("123");
      expect(result.status).toBe(Status.SUCCEEDED);
      expect(mockedAxios.post).toHaveBeenCalledTimes(1);
    });

    it("should handle upload errors", async () => {
      // Mock error response
      mockedAxios.post.mockRejectedValueOnce(new Error("Upload failed"));

      await expect(chunkr.upload(Buffer.from("test"))).rejects.toThrow(
        "Upload failed",
      );
    });
  });
});
