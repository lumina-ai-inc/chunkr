import axiosInstance from "./axios.config";

export async function createSetupIntent(accessToken: string) {
  try {
    const response = await axiosInstance.get(
      "/api/stripe/create-setup-intent",
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    return response.data.clientSecret; // Return the client secret directly
  } catch (error) {
    console.error("Error creating setup intent:", error);
    throw error;
  }
}
