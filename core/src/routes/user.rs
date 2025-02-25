use crate::configs::email_config;
use crate::models::chunkr::auth::UserInfo;
use crate::utils::routes::{create_user::create_user, get_user::get_user};
use crate::utils::services::email::EmailService;
use actix_web::{web, Error, HttpResponse};

pub async fn get_or_create_user(user_info: web::ReqData<UserInfo>) -> Result<HttpResponse, Error> {
    let user_info = user_info.into_inner();
    let user_id = user_info.clone().user_id;

    let user = match get_user(user_id).await {
        Ok(user) => user,
        Err(e) => {
            if e.to_string().contains("not found") {
                let user = create_user(user_info.clone()).await?;

                if let (Some(first_name), Some(email)) = (user_info.first_name, user_info.email) {
                    let email_config = email_config::Config::from_env()
                        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
                    let email_service = EmailService::new(email_config);

                    if let Err(e) = email_service.send_welcome_email(&first_name, &email).await {
                        log::error!("Failed to send welcome email: {}", e);
                    }
                }

                return Ok(HttpResponse::Ok().json(user));
            } else {
                return Err(e.into());
            }
        }
    };

    Ok(HttpResponse::Ok().json(user))
}
