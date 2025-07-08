use crate::configs::cal_config::Config;
use crate::configs::postgres_config::Client;
use crate::models::auth::UserInfo;
use crate::models::cal::{Attendee, CalBookingRequest, CalBookingResponse, OnboardingRequest};
use crate::models::user::{Information, Status};
use crate::utils::clients;
use crate::utils::clients::get_pg_client;
use actix_web::{web, Error, HttpResponse};

pub async fn create_onboarding(
    request: web::Json<OnboardingRequest>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let cal_config = Config::from_env().map_err(|_| {
        println!("cal.com config not set");
        actix_web::error::ErrorInternalServerError("cal.com configuration not properly set")
    })?;

    // Save the Information to the database
    let mut client: Client = get_pg_client().await.map_err(|e| {
        println!("Database connection error: {e}");
        actix_web::error::ErrorInternalServerError("Database connection error".to_string())
    })?;
    let transaction = client.transaction().await.map_err(|e| {
        println!("Transaction error: {e}");
        actix_web::error::ErrorInternalServerError("Transaction error".to_string())
    })?;

    let information = Information {
        use_case: request.information.use_case.clone(),
        usage: request.information.usage.clone(),
        file_types: request.information.file_types.clone(),
        referral_source: request.information.referral_source.clone(),
        add_ons: request.information.add_ons.clone(),
    };

    let onboarding_query = r#"
    UPDATE onboarding_records 
    SET information = $1, status = $2
    WHERE user_id = $3
    "#;

    transaction
        .execute(
            onboarding_query,
            &[
                &serde_json::to_value(&information)?,
                &Status::Completed,
                &user_info.user_id,
            ],
        )
        .await
        .map_err(|e| {
            println!("Database update error: {e}");
            actix_web::error::ErrorInternalServerError("Database update error".to_string())
        })?;
    transaction.commit().await.map_err(|e| {
        println!("Transaction commit error: {e}");
        actix_web::error::ErrorInternalServerError("Transaction commit error".to_string())
    })?;

    let email = user_info.email.clone().unwrap_or("".to_string());
    let name = user_info.first_name.clone().unwrap_or("".to_string());
    if !email.is_empty() && !name.is_empty() {
        let mut booking_fields_responses = std::collections::HashMap::new();

        let notes = format!(
            "Use Case: {}\nUsage: {}\nFile Types: {}\nReferral Source: {}\nAdd-ons: {}",
            information.use_case,
            information.usage,
            information.file_types,
            information.referral_source,
            information.add_ons.join(", ")
        );
        booking_fields_responses.insert("notes".to_string(), serde_json::Value::String(notes));

        let cal_booking_request = CalBookingRequest {
            start: request.start.clone(),
            attendee: Attendee {
                name,
                email,
                timezone: request.timezone.clone(),
            },
            event_type_id: cal_config.event_type_id.parse::<i32>().unwrap(),
            booking_fields_responses: Some(booking_fields_responses),
        };
        let client = clients::get_reqwest_client();

        let response = client
            .post(format!("{}/bookings", cal_config.url))
            .header("Authorization", format!("Bearer {}", cal_config.api_key))
            .header("cal-api-version", cal_config.booking_api_version)
            .header("Content-Type", "application/json")
            .json(&cal_booking_request)
            .send()
            .await;

        match response {
            Ok(resp) => match resp.json::<CalBookingResponse>().await {
                Ok(booking_response) => Ok(HttpResponse::Ok().json(booking_response)),
                Err(_) => Ok(
                    HttpResponse::InternalServerError().body("Failed to parse cal.com response")
                ),
            },
            Err(e) => Ok(HttpResponse::InternalServerError().body(e.to_string())),
        }
    } else {
        println!("User email or first name is none");
        Err(actix_web::error::ErrorInternalServerError(
            "User email or first name is none",
        ))
    }
}
