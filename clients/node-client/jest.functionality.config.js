module.exports = {
  // Uses ts-jest to handle TypeScript files
  preset: "ts-jest",

  // Runs tests in Node.js environment (not browser)
  testEnvironment: "node",

  // Runs functionality test files
  testMatch: ["**/__tests__/**/*.functionality.test.ts"],

  // Increasing timeout to 2 minutes since API processing can take longer
  testTimeout: 220000, // 220 seconds = 3 minutes and 40 seconds
};
