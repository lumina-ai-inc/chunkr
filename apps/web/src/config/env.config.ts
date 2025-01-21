export interface Features {
  pipeline: boolean;
  // Add new feature flags here
  // example: betaFeature: boolean;
}

export interface EnvConfig {
  features: Features;
}

export const getEnvConfig = (): EnvConfig => {
  return {
    features: {
      pipeline: import.meta.env.VITE_FEATURE_FLAG_PIPELINE === "true",
      // Add new feature implementations here
    },
  };
};

export function validateEnvConfig(): void {
  const requiredFlags: Array<keyof Features> = ["pipeline"];

  for (const flag of requiredFlags) {
    const value = import.meta.env[`VITE_FEATURE_FLAG_${flag.toUpperCase()}`];
    if (value !== "true" && value !== "false") {
      throw new Error(
        `VITE_FEATURE_FLAG_${flag.toUpperCase()} must be either "true" or "false"`,
      );
    }
  }
}

// Type helper for feature-guarded types
export type WhenEnabled<
  Flag extends keyof Features,
  T,
> = Features[Flag] extends true ? T | undefined : undefined;
