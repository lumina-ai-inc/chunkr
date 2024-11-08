use crate::models::server::user::{Discount, InvoiceStatus, Tier, UsageLimit, UsageType, User};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use serde_json::Value;
use std::str::FromStr;
pub async fn get_user(user_id: String, pool: &Pool) -> Result<User, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let query = r#"
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
        u.task_count,
        COALESCE(json_agg(json_build_object('usage_type', d.usage_type, 'amount', d.amount))::text, '[]') as discounts
    FROM 
        users u
    LEFT JOIN 
        api_keys ak ON u.user_id = ak.user_id
    LEFT JOIN 
        discounts d ON u.user_id = d.user_id AND d.usage_type IN ('Fast', 'HighQuality', 'Segment')
    WHERE 
        u.user_id = $1
    GROUP BY 
        u.user_id;
    "#;

    let row = client.query_opt(query, &[&user_id]).await?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("User with id {} not found", user_id),
        )
    })?;
    let discounts_json: String = row.get("discounts");
    let discounts: Vec<Discount> = serde_json::from_str::<Vec<Value>>(&discounts_json)?
        .into_iter()
        .filter(|d| d["usage_type"] != Value::Null && d["amount"] != Value::Null)
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();

    // Get usage limits from database
    let usage_limits = client
        .query(
            "SELECT usage_type, usage_limit FROM usage WHERE user_id = $1",
            &[&user_id],
        )
        .await?;

    let mut usage_map = std::collections::HashMap::new();
    for usage_row in usage_limits {
        let usage_type: String = usage_row.get("usage_type");
        let limit: i32 = usage_row.get("usage_limit");
        usage_map.insert(usage_type, limit);
    }

    let user = User {
        user_id: row.get("user_id"),
        customer_id: row.get("customer_id"),
        email: row.get("email"),
        first_name: row.get("first_name"),
        last_name: row.get("last_name"),
        api_keys: row.get("api_keys"),
        tier: row
            .get::<_, Option<String>>("tier")
            .and_then(|t| Tier::from_str(&t).ok())
            .unwrap_or(Tier::Free),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        usage: vec![
            UsageLimit {
                usage_type: UsageType::Fast,
                usage_limit: *usage_map.get("Fast").unwrap_or(&0),
                discounts: if discounts.is_empty() {
                    None
                } else {
                    Some(
                        discounts
                            .clone()
                            .into_iter()
                            .filter(|d| d.usage_type == UsageType::Fast)
                            .collect(),
                    )
                },
            },
            UsageLimit {
                usage_type: UsageType::HighQuality,
                usage_limit: *usage_map.get("HighQuality").unwrap_or(&0),
                discounts: if discounts.is_empty() {
                    None
                } else {
                    Some(
                        discounts
                            .clone()
                            .into_iter()
                            .filter(|d| d.usage_type == UsageType::HighQuality)
                            .collect(),
                    )
                },
            },
            UsageLimit {
                usage_type: UsageType::Segment,
                usage_limit: *usage_map.get("Segment").unwrap_or(&0),
                discounts: if discounts.is_empty() {
                    None
                } else {
                    Some(
                        discounts
                            .into_iter()
                            .filter(|d| d.usage_type == UsageType::Segment)
                            .collect(),
                    )
                },
            },
        ],
        task_count: row.get("task_count"),
    };

    Ok(user)
}

use serde::Serialize;

#[derive(Serialize)]
pub struct InvoiceSummary {
    pub invoice_id: String,
    pub status: InvoiceStatus,
    pub date_created: Option<chrono::NaiveDateTime>,
    pub amount_due: f32, // Added amount_due field
}

#[derive(Serialize, Clone)]
pub struct MonthlyUsage {
    pub month: String,
    pub total_cost: f32,
    pub usage_details: Vec<UsageDetail>,
}

#[derive(Serialize, Clone)]
pub struct UsageDetail {
    pub usage_type: String,
    pub count: i64,
    pub cost: f32,
}

pub async fn get_monthly_usage_count(
    user_id: String,
    pool: &Pool,
) -> Result<Vec<MonthlyUsage>, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    // Check user tier
    let tier_query = "SELECT tier FROM users WHERE user_id = $1";
    let user_tier: Option<String> = client.query_one(tier_query, &[&user_id]).await?.get(0);
    let query = r#"
        SELECT
            to_char(created_at, 'YYYY-MM') AS month,
            configuration::JSONB->>'model' AS usage_type,
            SUM(page_count) AS total_pages
        FROM
            tasks
        WHERE
            user_id = $1
            AND status = 'Succeeded'
        GROUP BY
            month, usage_type
        ORDER BY
            month DESC, usage_type;
        "#;

    let rows = client.query(query, &[&user_id]).await?;

    let mut monthly_usage_map: std::collections::HashMap<String, MonthlyUsage> =
        std::collections::HashMap::new();

    for row in rows {
        let month: String = row.get("month");
        let usage_type: String = row.get("usage_type");
        let total_pages: i64 = row.get("total_pages");

        let (_, cost) = if user_tier.as_deref() == Some("Free") {
            (0.0, 0.0) // Hardcode cost per type and total cost to 0 for Free tier
        } else {
            match usage_type.as_str() {
                "Fast" => (0.005, (total_pages as f64) * 0.005),
                "HighQuality" => (0.01, (total_pages as f64) * 0.01),
                "Segment" => (0.01, (total_pages as f64) * 0.01),
                _ => (0.0, 0.0),
            }
        };

        monthly_usage_map
            .entry(month.clone())
            .or_insert(MonthlyUsage {
                month: month.clone(),
                total_cost: 0.0,
                usage_details: Vec::new(),
            })
            .total_cost += cost as f32;

        monthly_usage_map
            .get_mut(&month)
            .unwrap()
            .usage_details
            .push(UsageDetail {
                usage_type,
                count: total_pages,
                cost: cost as f32,
            });
    }

    let monthly_usage_counts: Vec<MonthlyUsage> =
        monthly_usage_map.into_iter().map(|(_, v)| v).collect();
    // Sort monthly_usage_counts in descending order by month
    let mut monthly_usage_counts = monthly_usage_counts.clone();
    monthly_usage_counts.sort_by(|a, b| b.month.cmp(&a.month));

    Ok(monthly_usage_counts)
}

#[derive(Serialize)]
pub struct TaskInvoice {
    pub task_id: String,
    pub usage_type: String, //enum
    pub pages: i32,
    pub cost: f32,
    pub created_at: chrono::NaiveDateTime,
}

#[derive(Serialize)]
pub struct InvoiceDetail {
    pub invoice_id: String,
    pub stripe_invoice_id: Option<String>, // New field for Stripe invoice ID
    pub invoice_status: Option<String>,    // New field for invoice status
    pub tasks: Vec<TaskInvoice>,
}

pub async fn get_invoices(
    user_id: String,
    pool: &Pool,
) -> Result<Vec<InvoiceDetail>, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let query = r#"
    SELECT 
        invoice_id,
        stripe_invoice_id, 
        invoice_status,      
        task_id,
        usage_type,
        pages,
        cost,
        created_at
    FROM 
        task_invoices
    WHERE 
        user_id = $1
    ORDER BY 
        created_at DESC;
    "#;

    let rows = client.query(query, &[&user_id]).await?;

    let mut invoices: Vec<InvoiceDetail> = Vec::new();

    for row in rows {
        let invoice_id: String = row.get("invoice_id");
        let stripe_invoice_id: Option<String> = row.get("stripe_invoice_id"); // Get Stripe invoice ID
        let invoice_status: Option<String> = row.get("invoice_status"); // Get invoice status
        let task_invoice = TaskInvoice {
            task_id: row.get("task_id"),
            usage_type: row.get("usage_type"),
            pages: row.get("pages"),
            cost: row.get("cost"),
            created_at: row.get("created_at"),
        };

        // Check if the invoice already exists in the vector
        if let Some(invoice) = invoices.iter_mut().find(|inv| inv.invoice_id == invoice_id) {
            invoice.tasks.push(task_invoice);
        } else {
            invoices.push(InvoiceDetail {
                invoice_id,
                stripe_invoice_id, 
                invoice_status,   
                tasks: vec![task_invoice],
            });
        }
    }

    Ok(invoices)
}

pub async fn get_invoice_information(
    invoice_id: String,
    pool: &Pool,
) -> Result<InvoiceDetail, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let query = r#"
    SELECT 
        task_id,
        usage_type,
        pages,
        cost,
        created_at,
        stripe_invoice_id,
        invoice_status
    FROM 
        task_invoices
    WHERE 
        invoice_id = $1;
    "#;

    let rows: Vec<tokio_postgres::Row> = client.query(query, &[&invoice_id]).await?;

    let tasks = rows
        .iter()
        .map(|row| TaskInvoice {
            task_id: row.get("task_id"),
            usage_type: row.get("usage_type"),
            pages: row.get("pages"),
            cost: row.get("cost"),
            created_at: row.get("created_at"),
        })
        .collect();

    let stripe_invoice_id: Option<String> =
        rows.first().and_then(|row| row.get("stripe_invoice_id"));
    let invoice_status: Option<String> = rows.first().and_then(|row| row.get("invoice_status"));

    Ok(InvoiceDetail {
        invoice_id,
        stripe_invoice_id: stripe_invoice_id.clone(),
        invoice_status: invoice_status.clone(),
        tasks,
    })
}
