use chrono::{DateTime, Utc};
use chunkmydocs::utils::db::deadpool_postgres::{Client, Pool};
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

pub fn main() -> std::io::Result<()> {
    actix_web::rt::System::new().block_on(async move {
        let pg_pool = chunkmydocs::utils::db::deadpool_postgres::create_pool();
        process_daily_invoices(&pg_pool)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(())
    })
}
