use crate::utils::configs::expiration_config::Config as ExpirationConfig;
use crate::utils::configs::stripe_config::Config as StripeConfig;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::stripe::invoicer::invoice;
use std::time::Duration;
use tokio::time;

pub fn run_expiration_job() {
    actix_web::rt::spawn(async move {
        let expiration_config = ExpirationConfig::from_env().unwrap();
        let interval = expiration_config.job_interval;
        let mut interval = time::interval(Duration::from_secs(interval));
        println!("Expiration interval: {:?}", interval.period());
        loop {
            interval.tick().await;
            println!("Processing expired tasks");
            todo!()
        }
    });
}

pub fn run_invoice_job(pg_pool: Pool) {
    let stripe_config = match StripeConfig::from_env() {
        Ok(config) => config,
        Err(_) => {
            return;
        }
    };
    actix_web::rt::spawn(async move {
        let interval = stripe_config.invoice_interval;
        let mut interval = time::interval(Duration::from_secs(interval));
        println!("Invoice interval: {:?}", interval.period());
        loop {
            interval.tick().await;
            println!("Processing daily invoices");
            if let Err(e) = invoice(&pg_pool, None).await {
                eprintln!("Error processing daily invoices: {}", e);
            }
        }
    });
}

pub fn init_jobs(pg_pool: Pool) {
    run_expiration_job();
    run_invoice_job(pg_pool);
}
