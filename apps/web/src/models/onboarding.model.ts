export type Status = "success" | "error";

export interface ErrorDetails {
  message: string;
  error: string;
  statusCode: number;
}

export interface ErrorObject {
  code: string;
  message: string;
  details: ErrorDetails;
}

export interface Slot {
  start: string;
}

export interface SlotsData {
  [key: string]: Slot[];
}

export interface CalSlotsResponse {
  data?: SlotsData;
  status: Status;
  timestamp?: string;
  path?: string;
  error?: ErrorObject;
}

export interface Host {
  id: number;
  name: string;
  email: string;
  username: string;
  timeZone: string;
}

export interface EventType {
  id: number;
  slug: string;
}

export interface BookingAttendee {
  name: string;
  email: string;
  timeZone: string;
  phoneNumber?: string;
  language?: string;
}

export interface CalBookingData {
  id: number;
  uid: string;
  title: string;
  description?: string;
  hosts: Host[];
  status: string;
  cancellationReason?: string;
  cancelledByEmail?: string;
  reschedulingReason?: string;
  rescheduledByEmail?: string;
  rescheduledFromUid?: string;
  start: string;
  end: string;
  duration: number;
  eventTypeId: number;
  eventType: EventType;
  meetingUrl?: string;
  location?: string;
  absentHost?: boolean;
  createdAt: string;
  updatedAt: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata?: Record<string, any>;
  rating?: number;
  icsUid?: string;
  attendees: BookingAttendee[];
  guests?: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  bookingFieldsResponses?: Record<string, any>;
}

export interface CalBookingResponse {
  status: Status;
  timestamp?: string;
  path?: string;
  data?: CalBookingData;
  error?: ErrorObject;
}
