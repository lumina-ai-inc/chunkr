use crate::configs::postgres_config::Client;
use crate::configs::stripe_config::Config as StripeConfig;
use crate::utils::clients::get_pg_client;
use chrono::Utc;
use reqwest::Client as ReqwestClient;

pub async fn invoice() -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;
    let today = Utc::now();

    let invoices = client
        .query(
            "SELECT i.invoice_id, i.invoice_status, i.bill_date
             FROM invoices i
             JOIN users u ON i.user_id = u.user_id
             WHERE i.invoice_status IN ('Ongoing', 'Failed') 
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
            create_and_send_invoice(&invoice_id).await?;
        }
    }
    Ok(())
}

pub async fn create_and_send_invoice(invoice_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;
    let query = "
        SELECT i.invoice_id, i.user_id, i.amount_due, i.total_pages, i.bill_date,
               u.customer_id, u.tier
        FROM invoices i
        JOIN users u ON i.user_id = u.user_id
        WHERE i.invoice_id = $1
        LIMIT 1
    ";
    let rows = client.query(query, &[&invoice_id]).await?;
    if rows.is_empty() {
        return Err("Invoice not found".into());
    }

    let row = &rows[0];
    let stripe_customer_id: String = row.get("customer_id");
    let amount_due: f64 = row.get("amount_due");
    // let user_tier: String = row.get("tier");

    let stripe_config = StripeConfig::from_env()?;
    let reqwest_client = ReqwestClient::new();

    let create_invoice = reqwest_client
        .post("https://api.stripe.com/v1/invoices")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&[
            ("customer", stripe_customer_id.as_str()),
            ("auto_advance", "true"),
            ("collection_method", "charge_automatically"),
        ])
        .send()
        .await?;

    if !create_invoice.status().is_success() {
        return Err(format!("Failed to create invoice: {}", create_invoice.text().await?).into());
    }

    let invoice: serde_json::Value = create_invoice.json().await?;
    let stripe_invoice_id = invoice["id"].as_str().unwrap();
    let cost_in_cents = (amount_due * 100.0).round() as i64;

    let add_item = reqwest_client
        .post("https://api.stripe.com/v1/invoiceitems")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&[
            ("customer", stripe_customer_id.as_str()),
            ("amount", &cost_in_cents.to_string()),
            ("currency", "usd"),
            ("invoice", stripe_invoice_id),
        ])
        .send()
        .await?;

    if !add_item.status().is_success() {
        return Err(format!("Failed to add item to invoice: {}", add_item.text().await?).into());
    }

    let finalize = reqwest_client
        .post(&format!(
            "https://api.stripe.com/v1/invoices/{}/finalize",
            stripe_invoice_id
        ))
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .send()
        .await?;

    if !finalize.status().is_success() {
        return Err(format!("Failed to finalize invoice: {}", finalize.text().await?).into());
    } else {
        client
            .execute(
                "UPDATE invoices SET stripe_invoice_id = $1, invoice_status = 'Executed' WHERE invoice_id = $2",
                &[&stripe_invoice_id, &invoice_id],
            )
            .await?;
    }
    Ok(())
}
