import axiosInstance from "./axios.config";

export async function createSetupIntent(accessToken: string) {
  try {
    const response = await axiosInstance.get(
      "/stripe/create-setup-intent",
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    return response.data.setup_intent.client_secret; // Return the client secret directly
  } catch (error) {
    console.error("Error creating setup intent:", error);
    throw error;
  }
}

export async function createCustomerSession(accessToken: string) {
  try {
    const response = await axiosInstance.get("/stripe/create-session", {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
    return response.data.client_secret;
  } catch (error) {
    console.error("Error creating customer session:", error);
    throw error;
  }
}

export async function getUserInvoices() {
  try {
    const response = await axiosInstance.get("/stripe/get-user-invoices");
    return response.data;
  } catch (error) {
    console.error("Error fetching user invoices:", error);
    throw error;
  }
}

export async function getInvoiceDetail(accessToken: string, invoiceId: string) {
  try {
    const response = await axiosInstance.get(
      `/stripe/get-invoice-detail/${invoiceId}`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching invoice detail:", error);
    throw error;
  }
}
