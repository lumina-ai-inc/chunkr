use crate::configs::postgres_config::Client;
use crate::configs::stripe_config::Config as StripeConfig;
use crate::utils::clients::get_pg_client;
use chrono::{DateTime, Utc};
use reqwest::Client as ReqwestClient;
use serde_json::json;
use tokio_postgres::Row;

pub async fn invoice() -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;
    let today = Utc::now();

    // Get all invoices that are not completed and are either ongoing or failed
    // and their bill_date matches today
    let invoices = client
        .query(
            "SELECT i.invoice_id, i.invoice_status, i.bill_date
             FROM invoices i
             JOIN users u ON i.user_id = u.user_id
             WHERE i.invoice_status IN ('Ongoing', 'Failed') 
             AND u.tier = 'PayAsYouGo'
             AND DATE(i.bill_date) = DATE($1)",
            &[&today],
        )
        .await?;

    for row in invoices {
        let invoice_id: String = row.get("invoice_id");
        let invoice_status: String = row.get("invoice_status");

        if invoice_status == "Failed" {
            continue;
        }

        if invoice_status == "Ongoing" {
            println!(
                "creating and sending invoice for invoice id {:?}",
                invoice_id
            );
            create_and_send_invoice(&invoice_id).await?;
        }
    }

    Ok(())
}

pub async fn create_and_send_invoice(invoice_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;

    // Query to get invoice details, customer info, and usage
    let query = "
        SELECT i.invoice_id, i.user_id, i.amount_due, i.total_pages, i.bill_date,
               u.customer_id,
               ti.usage_type, SUM(ti.pages) as pages
        FROM invoices i
        JOIN users u ON i.user_id = u.user_id
        JOIN task_invoices ti ON i.invoice_id = ti.invoice_id
        WHERE i.invoice_id = $1
        GROUP BY i.invoice_id, i.user_id, i.amount_due, i.total_pages, 
                 i.bill_date, u.customer_id, ti.usage_type
    ";

    let rows = client.query(query, &[&invoice_id]).await?;

    if rows.is_empty() {
        return Err("Invoice not found".into());
    }

    let row = &rows[0];
    let stripe_customer_id: String = row.get("customer_id");
    let bill_date: DateTime<Utc> = row.get("bill_date");

    // Create line items based on usage
    let mut line_items = Vec::new();
    let mut discount_message = String::new();
    let mut discount_updates = Vec::new();
    let mut usage_summary = String::from(format!(
        "Billing period ending: {}\n\n",
        bill_date.format("%Y-%m-%d")
    ));

    for row in &rows {
        let usage_type: String = row.get("usage_type");
        let pages: i32 = row.get::<_, i64>("pages") as i32;

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

        let (price_id, _) = match usage_type.as_str() {
            "Fast" => (StripeConfig::from_env()?.page_fast_price_id, pages),
            "HighQuality" => (StripeConfig::from_env()?.page_high_quality_price_id, pages),
            "Segment" => (StripeConfig::from_env()?.segment_price_id, pages),
            _ => return Err("Invalid usage type".into()),
        };

        let mut charged_pages = pages;
        let mut discounted_pages = 0;

        // Apply discount if available
        if let Some(discount) = discount_amount {
            if discount > 0.0 {
                if discount >= pages as f64 {
                    discounted_pages = pages;
                    charged_pages = 0;
                } else {
                    discounted_pages = discount as i32;
                    charged_pages = pages - discounted_pages;
                }
                let remaining_discount = discount - discounted_pages as f64;
                discount_updates.push((
                    row.get::<_, String>("user_id"),
                    usage_type.clone(),
                    remaining_discount,
                ));
            }
        }

        if charged_pages > 0 {
            line_items.push(json!({
                "price": price_id,
                "quantity": charged_pages,
            }));
        }

        usage_summary.push_str(&format!("{} pages: {}\n", usage_type, pages));
        if discounted_pages > 0 {
            usage_summary.push_str(&format!(
                "discounted {} pages: {}\n",
                usage_type, discounted_pages
            ));
        }
        if charged_pages > 0 {
            usage_summary.push_str(&format!(
                "Pages charged {}: {}\n",
                usage_type, charged_pages
            ));
        }
    }

    discount_message.push_str(&usage_summary);
    println!("line items {:?}", line_items);
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

    for line_item in line_items {
        let response = reqwest_client
            .post("https://api.stripe.com/v1/invoiceitems")
            .header("Authorization", format!("Bearer {}", stripe_config.api_key))
            .form(&[
                ("customer", stripe_customer_id.as_str()),
                ("price", line_item["price"].as_str().unwrap()),
                ("quantity", &line_item["quantity"].to_string()),
                ("invoice", stripe_invoice_id), // Attach the item to the invoice
            ])
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!(
                "Failed to add invoice item to Stripe invoice: {}",
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
    } else {
        println!("successfully finalized invoice");
        // Update local invoice with Stripe invoice ID
        client
    .execute(
        "UPDATE invoices SET stripe_invoice_id = $1, invoice_status = 'Executed' WHERE invoice_id = $2",
        &[&stripe_invoice_id, &invoice_id],
            )
            .await?;
        // Update discounts table
        for (user_id, usage_type, remaining_discount) in discount_updates {
            client
                .execute(
                    "UPDATE discounts SET amount = $1 WHERE user_id = $2 AND usage_type = $3",
                    &[&remaining_discount, &user_id, &usage_type],
                )
                .await?;
        }
    }

    Ok(())
}
