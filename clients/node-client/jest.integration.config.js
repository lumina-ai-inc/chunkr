module.exports = {
  // Uses ts-jest to handle TypeScript files
  preset: "ts-jest",

  // Runs tests in Node.js environment (not browser)
  testEnvironment: "node",

  // Only runs files that end with .integration.test.ts
  testMatch: ["**/__tests__/**/*.integration.test.ts"],

  // Sets timeout to 30 seconds instead of Jest's default 5 seconds
  // because API calls take longer than regular unit tests
  testTimeout: 30000,
};
