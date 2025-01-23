module.exports = {
  // Uses ts-jest to handle TypeScript files
  preset: "ts-jest",

  // Runs tests in Node.js environment (not browser)
  testEnvironment: "node",

  // Runs functionality test files
  testMatch: ["**/__tests__/**/*.functionality.test.ts"],

  testTimeout: 2220000, // 2220 seconds = 37 minutes
};
