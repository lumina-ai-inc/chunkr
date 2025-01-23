use crate::configs::postgres_config::Client;
use crate::models::chunkr::user::{ InvoiceStatus, Tier, UsageLimit, UsageType, User};
use crate::utils::clients::get_pg_client;
use std::str::FromStr;

pub async fn get_user(user_id: String) -> Result<User, Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;
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
        s.last_paid_status
    FROM 
        users u
    LEFT JOIN 
        api_keys ak ON u.user_id = ak.user_id
    LEFT JOIN
        subscriptions s ON u.user_id = s.user_id
    WHERE 
        u.user_id = $1
    GROUP BY 
        u.user_id,
        s.last_paid_status;
    "#;

    let row = client.query_opt(query, &[&user_id]).await?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("User with id {} not found", user_id),
        )
    })?;

    let usage_limits = client
        .query(
            "SELECT usage_type, usage_limit, overage_usage FROM monthly_usage WHERE user_id = $1",
            &[&user_id],
        )
        .await?;

    let mut usage = Vec::new();
    for usage_row in usage_limits {
        let usage_type: String = usage_row.get("usage_type");
        let overage_usage: i32 = usage_row.get("overage_usage");
        let usage_limit: i32 = usage_row.get("usage_limit");


        usage.push(UsageLimit {
            usage_type: UsageType::from_str(&usage_type).unwrap_or(UsageType::Page),
            usage_limit,
            overage_usage,
        });
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
        usage,
        task_count: row.get("task_count"),
        last_paid_status: row.get("last_paid_status")
    };

    Ok(user)
}

use serde::Serialize;

#[derive(Serialize)]
pub struct InvoiceSummary {
    pub invoice_id: String,
    pub status: InvoiceStatus,
    pub date_created: Option<chrono::NaiveDateTime>,
    pub amount_due: f32,
}

#[derive(Serialize, Clone)]
pub struct MonthlyUsage {
    pub user_id: String,
    pub email: Option<String>,
    pub last_paid_status: Option<String>,
    pub month: String,
    pub total_cost: f64,
    pub usage_limit: i32,
    pub usage: Option<i32>,
    pub overage_usage: i32,
    pub tier: String,
}

pub async fn get_monthly_usage_count(
    user_id: String,
) -> Result<Vec<MonthlyUsage>, Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;

    let query = r#"
        SELECT 
            u.user_id,
            u.email,
            u.invoice_status,
            to_char(mu.created_at, 'YYYY-MM') as month,
            mu.usage,
            mu.usage_limit,
            mu.overage_usage,
            mu.tier,
            CAST(t.price_per_month + (mu.overage_usage * t.overage_rate) AS DOUBLE PRECISION) as total_cost
        FROM users u
        LEFT JOIN monthly_usage mu ON u.user_id = mu.user_id
        LEFT JOIN tiers t ON mu.tier = t.tier
        WHERE u.user_id = $1
        ORDER BY mu.created_at DESC
    "#;

    let rows = client.query(query, &[&user_id]).await?;

    let mut monthly_usage: Vec<MonthlyUsage> = Vec::new();

    for row in rows {
        monthly_usage.push(MonthlyUsage {
            user_id: row.get("user_id"),
            email: row.get("email"),
            last_paid_status: row.get("invoice_status"),
            month: row.get("month"),
            total_cost: row.get::<_, f64>("total_cost"),
            usage_limit: row.get("usage_limit"),
            usage: row.get("usage"),
            overage_usage: row.get("overage_usage"),
            tier: row.get("tier"),
        });
    }

    Ok(monthly_usage)
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
) -> Result<Vec<InvoiceDetail>, Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;

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
) -> Result<InvoiceDetail, Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;

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
