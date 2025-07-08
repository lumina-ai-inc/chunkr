use crate::configs::cal_config::Config;
use crate::models::cal::{CalSlotsQuery, CalSlotsResponse, SlotsQuery};
use crate::utils::clients;
use actix_web::{web, Error, HttpResponse};

pub async fn get_slots(query: web::Query<SlotsQuery>) -> Result<HttpResponse, Error> {
    let config = Config::from_env().map_err(|_| {
        actix_web::error::ErrorInternalServerError("Cal.com configuration not properly set")
    })?;

    let cal_query = CalSlotsQuery {
        start: query.start.clone(),
        end: query.end.clone(),
        event_type_id: config.event_type_id.clone(),
    };

    let client = clients::get_reqwest_client();

    let response = client
        .get(format!("{}/slots", config.url))
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("cal-api-version", config.slots_api_version)
        .query(&cal_query)
        .send()
        .await;

    match response {
        Ok(resp) => match resp.json::<CalSlotsResponse>().await {
            Ok(slots_response) => Ok(HttpResponse::Ok().json(slots_response)),
            Err(_) => {
                Ok(HttpResponse::InternalServerError().body("Failed to parse cal.com response"))
            }
        },
        Err(e) => Ok(HttpResponse::InternalServerError().body(e.to_string())),
    }
}
