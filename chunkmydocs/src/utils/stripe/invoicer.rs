use chrono::Datelike;
use chrono::NaiveDate;
use chrono::{DateTime, Utc};
// use chunkmydocs::models::server::user::InvoiceStatus;
use crate::utils::configs::stripe_config::Config as StripeConfig;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use reqwest::Client as ReqwestClient;
use serde_json::json;
use tokio_postgres::Row;
use uuid::Uuid;

pub async fn invoice(
    pool: &Pool,
    end_of_month: Option<NaiveDate>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    // Determine the end of the month date
    let today = Utc::now().naive_utc();
    let end_of_month_date = match end_of_month {
        Some(date) => date,
        None => {
            let (_, year) = today.year_ce();
            let month = today.month();
            let last_day = match month {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 => {
                    if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                        29
                    } else {
                        28
                    }
                }
                _ => unreachable!(),
            };
            NaiveDate::from_ymd_opt(year as i32, month, last_day).unwrap()
        }
    };

    // Get all invoices that are not completed and are either ongoing or failed
    let invoices = client
        .query(
            "SELECT i.invoice_id, i.invoice_status, MIN(ti.created_at) as oldest_task_date
             FROM invoices i
             JOIN task_invoices ti ON i.invoice_id = ti.invoice_id
             JOIN users u ON i.user_id = u.user_id
             WHERE i.invoice_status IN ('ongoing', 'failed') AND u.tier != 'free'
             GROUP BY i.invoice_id, i.invoice_status",
            &[],
        )
        .await?;

    for row in invoices {
        let invoice_id: String = row.get("invoice_id");
        let invoice_status: String = row.get("invoice_status");

        if invoice_status == "failed" {
            // Skip failed invoices
            continue;
        }

        // Check if today is the end of the month
        if invoice_status == "ongoing" && today.date() == end_of_month_date {
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
        JOIN users u ON i.user_id = u.user_id
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

    // Create line items based on usage
    let mut line_items = Vec::new();
    let mut discount_message = String::new();
    for row in &rows {
        let usage_type: String = row.get("usage_type");
        let pages: i32 = row.get("pages");

        // Check for discounts
        let discount_query = "
            SELECT amount FROM discounts 
            WHERE user_id = $1 AND usage_type = $2
        ";
        let discount_amount: Option<f64> = client
            .query_one(
                discount_query,
                &[&row.get::<_, String>("user_id"), &usage_type],
            )
            .await
            .ok()
            .map(|r: Row| r.get("amount"));

        let (price_id, quantity) = match usage_type.as_str() {
            "Fast" => (StripeConfig::from_env()?.page_fast_price_id, pages),
            "HighQuality" => (StripeConfig::from_env()?.page_high_quality_price_id, pages),
            "Segment" => (StripeConfig::from_env()?.segment_price_id, pages),
            _ => return Err("Invalid usage type".into()),
        };

        // Apply discount if available
        if let Some(_) = discount_amount {
            // let _discounted_price: f64 =
            //     (pages as f64 * amount).min(row.get::<_, f64>("amount_due"));
            line_items.push(json!({
                "price": price_id,
                "quantity": quantity,
            }));
            discount_message.push_str(&format!(
                "DISCOUNT APPLIED for USAGE TYPE: {} and {} PAGES\n",
                usage_type, pages
            ));
            // Remove the discount from the discounts table
            client
                .execute(
                    "DELETE FROM discounts WHERE user_id = $1 AND usage_type = $2",
                    &[&row.get::<_, String>("user_id"), &usage_type],
                )
                .await?;
        } else {
            line_items.push(json!({
                "price": price_id,
                "quantity": quantity,
            }));
        }
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
            ("description", &discount_message), // Add discount message to invoice
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
            "UPDATE invoices SET stripe_invoice_id = $1, invoice_status = 'Executed' WHERE invoice_id = $2",
            &[&stripe_invoice_id, &invoice_id],
        )
        .await?;

    Ok(())
}
