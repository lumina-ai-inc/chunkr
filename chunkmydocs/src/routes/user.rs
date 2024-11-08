use crate::models::server::auth::UserInfo;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::server::{ get_user::get_user, create_user::create_user };
use actix_web::{ web, HttpResponse, Error };

pub async fn get_or_create_user(
    user_info: web::ReqData<UserInfo>,
    pool: web::Data<Pool>
) -> Result<HttpResponse, Error> {
    let user_info = user_info.into_inner();
    let user_id = user_info.clone().user_id;

    let user = match get_user(user_id, &pool).await {
        Ok(user) => user,
        Err(e) => {
            if e.to_string().contains("not found") {
                let user = create_user(user_info, &pool).await?;
                return Ok(HttpResponse::Ok().json(user));
            } else {
                return Err(e.into());
            }
        }
    };

    Ok(HttpResponse::Ok().json(user))
}


