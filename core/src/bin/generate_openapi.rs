use core::generate_and_save_openapi_spec;
use clap::{Command, Arg};
use std::path::Path;
use std::env;

fn main() -> std::io::Result<()> {
    let matches = Command::new("OpenAPI Generator")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Chunkr")
        .about("Generates OpenAPI specification from Chunkr API")
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Sets the output file path (optional)")
                .required(false)
        )
        .get_matches();

    let output_path = if let Some(output) = matches.get_one::<String>("output") {
        output.to_string()
    } else {
        let current_dir = env::current_dir().expect("Could not determine current directory");
        let default_dir = current_dir.join("../.chunkr");
        default_dir.join("openapi.json").to_string_lossy().to_string()
    };
    
    if let Some(parent) = Path::new(&output_path).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
            println!("Created directory: {:?}", parent);
        }
    }
    
    println!("Generating OpenAPI specification to: {}", output_path);
    generate_and_save_openapi_spec(&output_path)?;
    println!("OpenAPI specification generated successfully!");
     
    Ok(())
}

