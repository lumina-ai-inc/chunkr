use crate::configs::expiration_config::Config as ExpirationConfig;
use crate::configs::stripe_config::Config as StripeConfig;
use crate::utils::jobs::expiration::expire;
use crate::utils::stripe::invoicer::invoice;
use std::time::Duration;
use tokio::time;

pub fn run_expiration_job() {
    actix_web::rt::spawn(async move {
        let expiration_config = ExpirationConfig::from_env().unwrap();
        let interval = expiration_config.job_interval;
        let mut interval = time::interval(Duration::from_secs(interval));
        loop {
            interval.tick().await;
            println!("Processing expired tasks");
            if let Err(e) = expire().await {
                eprintln!("Error processing expired tasks: {}", e);
            }
        }
    });
}

pub fn run_invoice_job() {
    let stripe_config = match StripeConfig::from_env() {
        Ok(config) => config,
        Err(_) => {
            return;
        }
    };
    actix_web::rt::spawn(async move {
        let interval = stripe_config.invoice_interval;
        let mut interval = time::interval(Duration::from_secs(interval));
        loop {
            interval.tick().await;
            println!("Processing daily invoices");
            if let Err(e) = invoice().await {
                eprintln!("Error processing daily invoices: {}", e);
            }
        }
    });
}

pub fn init_jobs() {
    run_expiration_job();
    run_invoice_job();
}
