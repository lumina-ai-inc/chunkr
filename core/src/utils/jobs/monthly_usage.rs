// use crate::utils::clients::get_pg_client;

// pub async fn maintain_monthly_usage() -> Result<(), Box<dyn std::error::Error>> {
//     let client = get_pg_client().await?;

//     client
//         .execute("SELECT maintain_monthly_usage_cron()", &[])
//         .await?;

//     Ok(())
// }
