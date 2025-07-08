use crate::models::user::Information;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Success,
    Error,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub message: String,
    pub error: String,
    #[serde(rename = "statusCode")]
    pub status_code: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorObject {
    pub code: String,
    pub message: String,
    pub details: ErrorDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SlotsQuery {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalSlotsQuery {
    pub start: String,
    pub end: String,
    #[serde(rename = "eventTypeId")]
    pub event_type_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Slot {
    pub start: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SlotsData {
    #[serde(flatten)]
    pub slots: IndexMap<String, Vec<Slot>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalSlotsResponse {
    pub data: Option<SlotsData>,
    pub status: Status,
    pub timestamp: Option<String>,
    pub path: Option<String>,
    pub error: Option<ErrorObject>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OnboardingRequest {
    pub start: String,
    pub timezone: String,
    pub information: Information,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Attendee {
    pub name: String,
    pub email: String,
    #[serde(rename = "timeZone")]
    pub timezone: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalBookingRequest {
    pub start: String,
    pub attendee: Attendee,
    #[serde(rename = "eventTypeId")]
    pub event_type_id: i32,
    #[serde(rename = "bookingFieldsResponses")]
    pub booking_fields_responses: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Host {
    pub id: i32,
    pub name: String,
    pub email: String,
    pub username: String,
    #[serde(rename = "timeZone")]
    pub time_zone: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EventType {
    pub id: i32,
    pub slug: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BookingAttendee {
    pub name: String,
    pub email: String,
    #[serde(rename = "timeZone")]
    pub time_zone: String,
    #[serde(rename = "phoneNumber")]
    pub phone_number: Option<String>,
    pub language: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalBookingData {
    pub id: i32,
    pub uid: String,
    pub title: String,
    pub description: Option<String>,
    pub hosts: Vec<Host>,
    pub status: String,
    #[serde(rename = "cancellationReason")]
    pub cancellation_reason: Option<String>,
    #[serde(rename = "cancelledByEmail")]
    pub cancelled_by_email: Option<String>,
    #[serde(rename = "reschedulingReason")]
    pub rescheduling_reason: Option<String>,
    #[serde(rename = "rescheduledByEmail")]
    pub rescheduled_by_email: Option<String>,
    #[serde(rename = "rescheduledFromUid")]
    pub rescheduled_from_uid: Option<String>,
    pub start: String,
    pub end: String,
    pub duration: i32,
    #[serde(rename = "eventTypeId")]
    pub event_type_id: i32,
    #[serde(rename = "eventType")]
    pub event_type: EventType,
    #[serde(rename = "meetingUrl")]
    pub meeting_url: Option<String>,
    pub location: Option<String>,
    #[serde(rename = "absentHost")]
    pub absent_host: Option<bool>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub rating: Option<i32>,
    #[serde(rename = "icsUid")]
    pub ics_uid: Option<String>,
    pub attendees: Vec<BookingAttendee>,
    pub guests: Option<Vec<String>>,
    #[serde(rename = "bookingFieldsResponses")]
    pub booking_fields_responses: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalBookingResponse {
    pub status: Status,
    pub timestamp: Option<String>,
    pub path: Option<String>,
    pub data: Option<CalBookingData>,
    pub error: Option<ErrorObject>,
}
