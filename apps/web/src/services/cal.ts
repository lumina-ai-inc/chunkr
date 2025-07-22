import axiosInstance from "./axios.config";
import {
  CalSlotsResponse,
  CalBookingResponse,
} from "../models/onboarding.model";

export async function getSlots(): Promise<CalSlotsResponse> {
  const start = new Date();
  const end = new Date();
  start.setDate(start.getDate() + 2);
  end.setDate(end.getDate() + 30);

  const startISO = start.toISOString();
  const endISO = end.toISOString();
  const { data } = await axiosInstance.get<CalSlotsResponse>(
    `/cal/slots?start=${startISO}&end=${endISO}`
  );
  return data;
}

export async function createOnboarding(
  start: string,
  timezone: string,
  onboardingData: {
    useCase: string;
    monthlyUsage: number;
    fileTypes: string;
    referralSource: string;
    addOns: string[];
  }
): Promise<CalBookingResponse> {
  const { data } = await axiosInstance.post<CalBookingResponse>(
    "/cal/onboarding",
    {
      start,
      timezone,
      information: {
        use_case: onboardingData.useCase,
        usage: onboardingData.monthlyUsage.toString(),
        file_types: onboardingData.fileTypes,
        referral_source: onboardingData.referralSource,
        add_ons: onboardingData.addOns,
      },
    }
  );
  return data;
}

export async function checkCalEndpoints(): Promise<boolean> {
  try {
    const start = new Date();
    const end = new Date();
    start.setDate(start.getDate() + 2);
    end.setDate(end.getDate() + 30);

    const startISO = start.toISOString();
    const endISO = end.toISOString();

    await axiosInstance.get(`/cal/slots?start=${startISO}&end=${endISO}`);
    return true;
  } catch {
    return false;
  }
}
