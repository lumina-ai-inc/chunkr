use chrono::{DateTime, Utc};
use chunkmydocs::utils::configs::stripe_config::Config as StripeConfig;
use chunkmydocs::utils::db::deadpool_postgres::{Client, Pool};
use reqwest::Client as ReqwestClient;
use serde_json::json;
use uuid::Uuid;

pub async fn process_daily_invoices(pool: &Pool) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    // Get all tasks that are not in task_invoices
    let tasks = client
        .query(
            "SELECT t.task_id, t.user_id, t.page_count, t.created_at, u.type as usage_type
         FROM TASKS t
         LEFT JOIN USAGE_TYPE u ON t.configuration = u.id
         WHERE t.task_id NOT IN (SELECT task_id FROM task_invoices)
         AND t.status = 'completed'",
            &[],
        )
        .await?;

    for row in tasks {
        let user_id: String = row.get("user_id");
        let task_id: String = row.get("task_id");
        let pages: i32 = row.get("page_count");
        let usage_type: String = row.get("usage_type");
        let created_at: DateTime<Utc> = row.get("created_at");

        // Check if there's an ongoing invoice for this user
        let ongoing_invoice = client
            .query_opt(
                "SELECT invoice_id
             FROM invoices
             WHERE user_id = $1 AND invoice_status = 'ongoing'",
                &[&user_id],
            )
            .await?;

        let invoice_id = match ongoing_invoice {
            Some(row) => row.get("invoice_id"),
            None => {
                // Create a new invoice
                let new_invoice_id = Uuid::new_v4().to_string();
                client.execute(
                    "INSERT INTO invoices (invoice_id, user_id, tasks, invoice_status, amount_due, total_pages)
                     VALUES ($1, $2, ARRAY[$3], 'ongoing', 0, 0)",
                    &[&new_invoice_id, &user_id, &task_id],
                ).await?;
                new_invoice_id
            }
        };

        // Get the cost per unit for this usage type
        let cost_per_unit: f32 = client
            .query_one(
                "SELECT cost_per_unit_dollars
             FROM USAGE_TYPE
             WHERE type = $1",
                &[&usage_type],
            )
            .await?
            .get("cost_per_unit_dollars");

        let cost = cost_per_unit * pages as f32;

        // Insert into task_invoices
        client.execute(
            "INSERT INTO task_invoices (task_id, invoice_id, usage_type, pages, cost, created_at)
             VALUES ($1, $2, $3, $4, $5, $6)",
            &[&task_id, &invoice_id, &usage_type, &pages, &cost, &created_at],
        ).await?;

        // Update the invoice
        client
            .execute(
                "UPDATE invoices
             SET tasks = array_append(tasks, $1),
                 amount_due = amount_due + $2,
                 total_pages = total_pages + $3
             WHERE invoice_id = $4",
                &[&task_id, &cost, &pages, &invoice_id],
            )
            .await?;
    }

    Ok(())
}

pub async fn invoices_to_execute(pool: &Pool) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    // Get all invoices that are not completed and are either ongoing or failed
    let invoices = client
        .query(
            "SELECT i.invoice_id, i.invoice_status, MIN(ti.created_at) as oldest_task_date
             FROM invoices i
             JOIN task_invoices ti ON i.invoice_id = ti.invoice_id
             WHERE i.invoice_status IN ('ongoing', 'failed')
             GROUP BY i.invoice_id, i.invoice_status",
            &[],
        )
        .await?;

    for row in invoices {
        let invoice_id: String = row.get("invoice_id");
        let invoice_status: String = row.get("invoice_status");
        let oldest_task_date: DateTime<Utc> = row.get("oldest_task_date");

        if invoice_status == "failed" {
            // Skip failed invoices
            continue;
        }

        // Check if the invoice has been ongoing for more than 30 days
        if invoice_status == "ongoing" && (Utc::now() - oldest_task_date).num_days() > 30 {
            create_and_send_invoice(&pool, &invoice_id).await?;
        }
    }

    Ok(())
}

pub async fn create_and_send_invoice(
    db_pool: &Pool,
    invoice_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = db_pool.get().await?;

    // Query to get invoice details, customer info, and usage
    let query = "
        SELECT i.invoice_id, i.user_id, i.amount_due, i.total_pages,
               u.stripe_customer_id,
               ti.usage_type, SUM(ti.pages) as pages
        FROM invoices i
        JOIN users u ON i.user_id = u.id
        JOIN task_invoices ti ON i.invoice_id = ti.invoice_id
        WHERE i.invoice_id = $1
        GROUP BY i.invoice_id, i.user_id, i.amount_due, i.total_pages, u.stripe_customer_id, ti.usage_type
    ";

    let rows = client.query(query, &[&invoice_id]).await?;

    if rows.is_empty() {
        return Err("Invoice not found".into());
    }

    let row = &rows[0];
    let stripe_customer_id: String = row.get("stripe_customer_id");
    let amount_due: f64 = row.get("amount_due");

    // Create line items based on usage
    let mut line_items = Vec::new();
    for row in &rows {
        let usage_type: String = row.get("usage_type");
        let pages: i32 = row.get("pages");

        let (price_id, quantity) = match usage_type.as_str() {
            "Fast" => (StripeConfig::from_env()?.page_fast_price_id, pages),
            "HighQuality" => (StripeConfig::from_env()?.page_high_quality_price_id, pages),
            "Segment" => (StripeConfig::from_env()?.segment_price_id, pages),
            _ => return Err("Invalid usage type".into()),
        };

        line_items.push(json!({
            "price": price_id,
            "quantity": quantity,
        }));
    }

    // Create invoice in Stripe
    let stripe_config = StripeConfig::from_env()?;
    let reqwest_client = ReqwestClient::new();
    let response = reqwest_client
        .post("https://api.stripe.com/v1/invoices")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&[
            ("customer", stripe_customer_id.as_str()),
            ("auto_advance", "true"),
            ("collection_method", "charge_automatically"),
        ])
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to create Stripe invoice: {}",
            response.text().await?
        )
        .into());
    }

    let invoice: serde_json::Value = response.json().await?;
    let stripe_invoice_id = invoice["id"].as_str().unwrap();

    // Add line items to the invoice
    for line_item in line_items {
        let response = reqwest_client
            .post(&format!(
                "https://api.stripe.com/v1/invoices/{}/lines",
                stripe_invoice_id
            ))
            .header("Authorization", format!("Bearer {}", stripe_config.api_key))
            .form(&[
                ("price", line_item["price"].as_str().unwrap()),
                ("quantity", &line_item["quantity"].to_string()),
            ])
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!(
                "Failed to add line item to Stripe invoice: {}",
                response.text().await?
            )
            .into());
        }
    }

    // Finalize the invoice
    let response = reqwest_client
        .post(&format!(
            "https://api.stripe.com/v1/invoices/{}/finalize",
            stripe_invoice_id
        ))
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to finalize Stripe invoice: {}",
            response.text().await?
        )
        .into());
    }

    // Update local invoice with Stripe invoice ID
    client
        .execute(
            "UPDATE invoices SET stripe_invoice_id = $1 WHERE invoice_id = $2",
            &[&stripe_invoice_id, &invoice_id],
        )
        .await?;

    Ok(())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let pg_pool = chunkmydocs::utils::db::deadpool_postgres::create_pool();
    process_daily_invoices(&pg_pool)
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    invoices_to_execute(&pg_pool)
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    Ok(())
}
