use crate::configs::postgres_config::Client;
use crate::configs::user_config::Config as UserConfig;
use crate::models::auth::UserInfo;
use crate::models::user::{Tier, UsageLimit, UsageType, User};
use crate::utils::clients::get_pg_client;
use prefixed_api_key::PrefixedApiKeyController;
use serde::{Deserialize, Serialize};

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

pub async fn create_user(user_info: UserInfo) -> Result<User, Box<dyn std::error::Error>> {
    let mut client: Client = get_pg_client().await?;
    let user_config = UserConfig::from_env().unwrap();

    let controller = PrefixedApiKeyController::configure()
        .prefix("ch".to_owned())
        .seam_defaults()
        .finalize()?;

    let (pak, _hash) = controller.generate_key_and_hash();
    let key = pak.to_string();

    let tier: Tier = match user_config.self_hosted {
        true => Tier::SelfHosted,
        false => Tier::Free,
    };

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

    let monthly_usage_query = r#"
    INSERT INTO monthly_usage (user_id, usage_type, usage, usage_limit, year, month, tier, overage_usage, billing_cycle_start, billing_cycle_end)
    SELECT $1, $2, $3, 
        CASE 
            WHEN EXISTS (SELECT 1 FROM pre_applied_free_pages p WHERE p.email = $5) 
            THEN (SELECT amount FROM pre_applied_free_pages p WHERE p.email = $5 LIMIT 1)
            ELSE t.usage_limit
        END,
        EXTRACT(YEAR FROM CURRENT_TIMESTAMP), 
        EXTRACT(MONTH FROM CURRENT_TIMESTAMP), 
        t.tier,
        0,
        CURRENT_DATE,
        (CURRENT_DATE + INTERVAL '30 days')::TIMESTAMPTZ
    FROM tiers t
    WHERE t.tier = $4
    "#;

    transaction
        .execute(
            monthly_usage_query,
            &[
                &user_info.user_id,
                &UsageType::Page.to_string(),
                &0i32,
                &tier.to_string(),
                &user_info.email,
            ],
        )
        .await?;

    let check_result = transaction
        .query_opt(
            "SELECT 1 FROM pre_applied_free_pages WHERE email = $1",
            &[&user_info.email],
        )
        .await?;

    if check_result.is_some() {
        let update_pre_applied_query = r#"
        UPDATE pre_applied_free_pages 
        SET consumed = TRUE,
            updated_at = CURRENT_TIMESTAMP
        WHERE email = $1
        "#;

        transaction
            .execute(update_pre_applied_query, &[&user_info.email])
            .await?;
    }

    transaction.commit().await?;

    let usage_limit = client
        .query_one(
            "SELECT usage_limit FROM monthly_usage WHERE user_id = $1 AND usage_type = $2 ORDER BY billing_cycle_start DESC LIMIT 1",
            &[&user_info.user_id, &UsageType::Page.to_string()],
        )
        .await?
        .get::<_, i32>("usage_limit");

    let user = User {
        user_id: user_row.get("user_id"),
        customer_id: user_row.get("customer_id"),
        email: user_row.get("email"),
        first_name: user_row.get("first_name"),
        last_name: user_row.get("last_name"),
        api_keys: vec![key],
        tier: user_row
            .get::<_, Option<String>>("tier")
            .map(|t| t.parse::<Tier>())
            .unwrap_or(Ok(Tier::Free))
            .unwrap_or(Tier::Free),
        created_at: user_row.get("created_at"),
        updated_at: user_row.get("updated_at"),
        usage: vec![UsageLimit {
            usage_type: UsageType::Page,
            usage_limit,
            overage_usage: 0,
        }],
        task_count: Some(0),
        last_paid_status: None,
    };

    Ok(user)
}
