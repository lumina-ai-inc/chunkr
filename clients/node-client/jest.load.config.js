module.exports = {
  // Uses ts-jest to handle TypeScript files
  preset: "ts-jest",

  // Runs tests in Node.js environment (not browser)
  testEnvironment: "node",

  // Runs both load and functionality test files
  testMatch: ["**/__tests__/**/*.load.test.ts"],

  testTimeout: 3600000, // 1 hour timeout for entire test suite
};
