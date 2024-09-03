use crate::models::auth::auth::UserInfo;
use crate::models::server::user::{ User, Tier, UsageType, Usage };
use crate::utils::db::deadpool_postgres::{ Client, Pool };
use actix_web::{ web, HttpResponse, Error };
use prefixed_api_key::PrefixedApiKeyController;
use std::collections::HashMap;
use chrono::Utc;
use std::str::FromStr;

/// Get User
///
/// Get user information, if its the first time the user is logging in, create a new user
#[utoipa::path(
    get,
    path = "/user",
    context_path = "",
    tag = "user",
    responses((
        status = 200,
        description = "Get user information, if it's the first time the user is logging in, create a new user",
        body = UserResponse,
    ))
)]
pub async fn get_or_create_user(
    user_info: web::ReqData<UserInfo>,
    pool: web::Data<Pool>
) -> Result<HttpResponse, Error> {
    let user_info = user_info.into_inner();
    let user_id = user_info.clone().user_id;

    let user = match get_user(user_id, &pool).await {
        Ok(user) => user,
        Err(e) => {
            println!("Error: {}", e);
            let user = create_user(user_info, &pool).await?;
            return Ok(HttpResponse::Ok().json(user));
        }
    };

    Ok(HttpResponse::Ok().json(user))
}

async fn get_user(user_id: String, pool: &Pool) -> Result<User, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let query =
        r#"
    SELECT 
        u.user_id,
        u.customer_id,
        u.email,
        u.first_name,
        u.last_name,
        array_agg(DISTINCT ak.key) as api_keys,
        u.tier,
        u.created_at,
        u.updated_at,
        json_agg(
            json_build_object(
                'usage', COALESCE(us.usage, 0),
                'usage_limit', COALESCE(us.usage_limit, 0),
                'usage_type', us.usage_type,
                'unit', us.unit,
                'created_at', us.created_at,
                'updated_at', us.created_at
            )
        )::text AS usages
    FROM 
        users u
    LEFT JOIN 
        USAGE us ON u.user_id = us.user_id
    LEFT JOIN
        API_KEYS ak ON u.user_id = ak.user_id
    WHERE 
        u.user_id = $1
    GROUP BY 
        u.user_id;
    "#;

    let row = client
        .query_opt(query, &[&user_id]).await?
        .ok_or_else(||
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("User with id {} not found", user_id)
            )
        )?;

    let user = User {
        user_id: row.get("user_id"),
        customer_id: row.get("customer_id"),
        email: row.get("email"),
        first_name: row.get("first_name"),
        last_name: row.get("last_name"),
        api_keys: row.get("api_keys"),
        tier: row.get::<_, Option<String>>("tier")
            .and_then(|t| Tier::from_str(&t).ok())
            .unwrap_or(Tier::Free),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        usages: serde_json::from_str(&row.get::<_, String>("usages")).unwrap_or_default(),
    };

    Ok(user)
}

async fn create_user(user_info: UserInfo, pool: &Pool) -> Result<User, Box<dyn std::error::Error>> {
    let mut client: Client = pool.get().await?;

    let controller = PrefixedApiKeyController::configure()
        .prefix("lu".to_owned())
        .seam_defaults()
        .finalize()?;

    let (pak, _hash) = controller.generate_key_and_hash();
    let key = pak.to_string();

    // Create usage limits
    let usage_limits: HashMap<UsageType, i32> = HashMap::from([
        (UsageType::Fast, 1000),
        (UsageType::HighQuality, 500),
        (UsageType::Segment, 250),
    ]);
    let usage_limits_clone = usage_limits.clone();

    let transaction = client.transaction().await?;

    // Insert into users table
    let user_query =
        r#"
    INSERT INTO users (user_id, email, first_name, last_name, tier)
    VALUES ($1, $2, $3, $4, $5)
    RETURNING *
    "#;

    let user_row = transaction.query_one(
        user_query,
        &[
            &user_info.user_id,
            &user_info.email,
            &user_info.first_name,
            &user_info.last_name,
            &Tier::Free.to_string(),
        ]
    ).await?;

    // Insert into api_keys table
    let api_key_query =
        r#"
    INSERT INTO api_keys (key, user_id, access_level, active)
    VALUES ($1, $2, $3, $4)
    "#;

    transaction.execute(api_key_query, &[&key, &user_info.user_id, &"user", &true]).await?;

    // Insert into USAGE table
    let usage_query =
        r#"
    INSERT INTO USAGE (user_id, usage, usage_limit, usage_type, unit)
    VALUES ($1, $2, $3, $4, $5)
    "#;

    for (usage_type, limit) in usage_limits {
        transaction.execute(
            usage_query,
            &[&user_info.user_id, &0i32, &limit, &usage_type.to_string(), &usage_type.get_unit()]
        ).await?;
    }

    transaction.commit().await?;

    // Construct and return the User object
    let user = User {
        user_id: user_row.get("user_id"),
        customer_id: user_row.get("customer_id"),
        email: user_row.get("email"),
        first_name: user_row.get("first_name"),
        last_name: user_row.get("last_name"),
        api_keys: vec![key],
        tier: user_row
            .get::<_, Option<String>>("tier")
            .and_then(|t| Tier::from_str(&t).ok())
            .unwrap_or(Tier::Free),
        created_at: user_row.get("created_at"),
        updated_at: user_row.get("updated_at"),
        usages: usage_limits_clone
            .into_iter()
            .map(|(usage_type, limit)| Usage {
                usage: 0,
                usage_limit: limit,
                usage_type: usage_type.to_string(),
                unit: usage_type.get_unit(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            })
            .collect(),
    };

    Ok(user)
}
