use crate::models::auth::auth::UserInfo;
use crate::models::server::user::{InvoiceStatus, Tier, Usage, UsageType, User};
use crate::utils::configs::user_config::Config as UserConfig;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use chrono::Utc;
use prefixed_api_key::PrefixedApiKeyController;
use std::collections::HashMap;
use std::str::FromStr;

pub async fn create_user(
    user_info: UserInfo,
    pool: &Pool,
) -> Result<User, Box<dyn std::error::Error>> {
    let mut client: Client = pool.get().await?;
    let user_config = UserConfig::from_env().unwrap();

    let controller = PrefixedApiKeyController::configure()
        .prefix("lu".to_owned())
        .seam_defaults()
        .finalize()?;

    let (pak, _hash) = controller.generate_key_and_hash();
    let key = pak.to_string();

    let tier: Tier = match user_config.self_hosted {
        true => Tier::SelfHosted,
        false => Tier::Free,
    };

    let usage_limits: HashMap<UsageType, i32> = HashMap::from([
        (UsageType::Fast, UsageType::Fast.get_usage_limit(&tier)),
        (
            UsageType::HighQuality,
            UsageType::HighQuality.get_usage_limit(&tier),
        ),
        (
            UsageType::Segment,
            UsageType::Segment.get_usage_limit(&tier),
        ),
    ]);

    let usage_limits_clone = usage_limits.clone();

    let transaction = client.transaction().await?;

    let user_query = r#"
    INSERT INTO users (user_id, email, first_name, last_name, tier)
    VALUES ($1, $2, $3, $4, $5)
    RETURNING *
    "#;

    let user_row = transaction
        .query_one(
            user_query,
            &[
                &user_info.user_id,
                &user_info.email,
                &user_info.first_name,
                &user_info.last_name,
                &tier.to_string(),
            ],
        )
        .await?;

    let api_key_query = r#"
    INSERT INTO api_keys (key, user_id, access_level, active)
    VALUES ($1, $2, $3, $4)
    "#;

    transaction
        .execute(api_key_query, &[&key, &user_info.user_id, &"admin", &true])
        .await?;

    let usage_query = r#"
    INSERT INTO USAGE (user_id, usage, usage_limit, usage_type, unit)
    VALUES ($1, $2, $3, $4, $5)
    "#;

    for (usage_type, limit) in usage_limits {
        transaction
            .execute(
                usage_query,
                &[
                    &user_info.user_id,
                    &0i32,
                    &limit,
                    &usage_type.to_string(),
                    &usage_type.get_unit(),
                ],
            )
            .await?;
    }

    transaction.commit().await?;

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
        invoice_status: Some(InvoiceStatus::NoInvoice),
    };

    Ok(user)
}
