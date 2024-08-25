use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=migrations");

    println!("cargo:rerun-if-changed=.env");

    Ok(())
}
