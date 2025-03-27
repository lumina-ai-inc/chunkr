use crate::models::chunkr::auth::UserInfo;
use crate::utils::routes::{create_user::create_user, get_user::get_user};
use actix_web::{web, Error, HttpResponse};

pub async fn get_or_create_user(user_info: web::ReqData<UserInfo>) -> Result<HttpResponse, Error> {
    let user_info = user_info.into_inner();
    let user_id = user_info.clone().user_id;

    let user = match get_user(user_id).await {
        Ok(user) => user,
        Err(e) => {
            if e.to_string().contains("not found") {
                let user = create_user(user_info).await?;
                return Ok(HttpResponse::Ok().json(user));
            } else {
                return Err(e.into());
            }
        }
    };

    Ok(HttpResponse::Ok().json(user))
}
