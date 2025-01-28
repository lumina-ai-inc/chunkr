import axiosInstance from "./axios.config";
import { StripeCheckoutSession } from "../models/stripe.models";

export async function createCheckoutSession(
  accessToken: string,
  tier: string
): Promise<StripeCheckoutSession> {
  try {
    const response = await axiosInstance.post(
      "/stripe/checkout",
      { tier },
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error creating checkout session:", error);
    throw error;
  }
}

export async function getCheckoutSession(
  accessToken: string,
  sessionId: string
): Promise<StripeCheckoutSession> {
  try {
    const response = await axiosInstance.get(`/stripe/checkout/${sessionId}`, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error getting checkout session:", error);
    throw error;
  }
}

export async function getBillingPortalSession(
  accessToken: string,
  customerId: string
) {
  try {
    const response = await axiosInstance.get(`/stripe/billing-portal`, {
      params: { customer_id: customerId },
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error getting billing portal session:", error);
    throw error;
  }
}
