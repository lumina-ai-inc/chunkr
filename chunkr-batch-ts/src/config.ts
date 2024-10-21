import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const INPUT_FOLDER = path.join(__dirname, "..", "input");
export const MAX_CONCURRENT_REQUESTS = 1000;
export const REQUESTS_PER_SECOND = 5;
export const MAX_FILES_TO_PROCESS = 20;
