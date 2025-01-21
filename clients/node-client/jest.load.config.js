module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/__tests__/**/*.load.test.ts"],
  testTimeout: 300000, // 5 minute timeout for load tests
};
