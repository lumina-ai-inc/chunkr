use crate::models::server::user::{InvoiceStatus, Tier, User};
use crate::utils::db::deadpool_postgres::{Client, Pool};
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
        u.updated_at
    FROM 
        users u
    LEFT JOIN 
        api_keys ak ON u.user_id = ak.user_id
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

#[derive(Serialize)]
pub struct MonthlyUsageCount {
    pub month: String,
    pub usage_type: String,
    pub count: i64,
}

pub async fn get_monthly_usage_count(
    user_id: String,
    pool: &Pool,
) -> Result<Vec<MonthlyUsageCount>, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let query = r#"
    SELECT 
        to_char(created_at, 'YYYY-MM') AS month,
        configuration::JSONB->>'model' AS usage_type,
        COUNT(*) AS count
    FROM 
        tasks
    WHERE 
        user_id = $1 
    GROUP BY 
        month, usage_type
    ORDER BY 
        month, usage_type;
    "#;

    let rows = client.query(query, &[&user_id]).await?;

    let monthly_usage_counts: Vec<MonthlyUsageCount> = rows
        .into_iter()
        .map(|row| MonthlyUsageCount {
            month: row.get("month"),
            usage_type: row.get("usage_type"),
            count: row.get("count"),
        })
        .collect();

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
                invoice_id: invoice_id,
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
        created_at
    FROM 
        task_invoices
    WHERE 
        invoice_id = $1;
    "#;

    let rows = client.query(query, &[&invoice_id]).await?;

    let tasks = rows
        .into_iter()
        .map(|row| TaskInvoice {
            task_id: row.get("task_id"),
            usage_type: row.get("usage_type"),
            pages: row.get("pages"),
            cost: row.get("cost"),
            created_at: row.get("created_at"),
        })
        .collect();

    Ok(InvoiceDetail { invoice_id, tasks })
}
