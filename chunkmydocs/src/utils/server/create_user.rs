use crate::models::server::auth::UserInfo;
use crate::models::server::user::{Tier, UsageLimit, UsageType, User};
use crate::utils::configs::user_config::Config as UserConfig;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use prefixed_api_key::PrefixedApiKeyController;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Serialize, Deserialize)]
struct PreAppliedPages {
    usage_type: String,
    amount: i32,
}

impl From<tokio_postgres::Row> for PreAppliedPages {
    fn from(row: tokio_postgres::Row) -> Self {
        Self {
            usage_type: row.get("usage_type"),
            amount: row.get("amount"),
        }
    }
}

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

    let transaction = client.transaction().await?;

    let mut usage_limits: HashMap<UsageType, i32> = HashMap::from([
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
    let check_query = r#"SELECT 1 FROM pre_applied_free_pages WHERE email = $1"#;
    let check_result = transaction
        .query_opt(check_query, &[&user_info.email])
        .await?;

    if check_result.is_some() {
        let pre_applied_discount_pages_query =
            r#"SELECT usage_type, amount FROM pre_applied_free_pages WHERE email = $1"#;

        let pre_applied_pages: Vec<PreAppliedPages> = transaction
            .query(pre_applied_discount_pages_query, &[&user_info.email])
            .await?
            .into_iter()
            .map(PreAppliedPages::from)
            .collect();

        for pre_applied_page in pre_applied_pages {
            if let Ok(usage_type) = UsageType::from_str(&pre_applied_page.usage_type) {
                usage_limits.insert(usage_type, pre_applied_page.amount);
            }
        }
    }

    transaction.commit().await?;

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

    for (usage_type, limit) in &usage_limits {
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
    if check_result.is_some() {
        let transaction2 = client.transaction().await?;
        let update_pre_applied_query = r#"
        UPDATE pre_applied_free_pages 
        SET consumed = TRUE,
            updated_at = CURRENT_TIMESTAMP
        WHERE email = $1
        "#;

        transaction2
            .execute(update_pre_applied_query, &[&user_info.email])
            .await?;
        transaction2.commit().await?;
    }

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
        usage: vec![
            UsageLimit {
                usage_type: UsageType::Fast,
                usage_limit: usage_limits.get(&UsageType::Fast).copied().unwrap_or(1000),
                discounts: None,
            },
            UsageLimit {
                usage_type: UsageType::HighQuality,
                usage_limit: usage_limits
                    .get(&UsageType::HighQuality)
                    .copied()
                    .unwrap_or(500),
                discounts: None,
            },
            UsageLimit {
                usage_type: UsageType::Segment,
                usage_limit: usage_limits
                    .get(&UsageType::Segment)
                    .copied()
                    .unwrap_or(250),
                discounts: None,
            },
        ],
        task_count: Some(0),
    };

    Ok(user)
}
