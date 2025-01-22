module.exports = {
  // Uses ts-jest to handle TypeScript files
  preset: "ts-jest",

  // Runs tests in Node.js environment (not browser)
  testEnvironment: "node",

  // Runs both load and functionality test files
  testMatch: ["**/__tests__/**/*.load.test.ts"],

  // Sets timeout to 30 seconds instead of Jest's default 5 seconds
  // because API calls take longer than regular unit tests
  testTimeout: 3600000, // 1 hour timeout
};
