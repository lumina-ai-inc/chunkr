use crate::utils::clients;

pub async fn timeout(timeout: u32) -> Result<(), Box<dyn std::error::Error>> {
    let client = clients::get_pg_client().await?;

    let rows_affected = client
        .execute(
            "UPDATE tasks SET status = 'Failed', message = 'Task timed out' 
             WHERE status = 'Processing' 
             AND (started_at IS NULL OR started_at < NOW() - INTERVAL '1 second' * $1::float8)",
            &[&(timeout as f64)],
        )
        .await?;

    println!("Timed out {} processing tasks", rows_affected);
    Ok(())
}
