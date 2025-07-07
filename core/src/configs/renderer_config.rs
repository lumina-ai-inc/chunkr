use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_headless")]
    pub headless: bool,
    #[serde(default = "default_sandbox")]
    pub sandbox: bool,
    pub chrome_path: Option<PathBuf>,
    #[serde(default = "default_chrome_cache_dir")]
    pub chrome_cache_dir: Option<PathBuf>,
    #[serde(default = "default_chrome_user_data_dir")]
    pub chrome_user_data_dir: Option<PathBuf>,
    #[serde(default = "default_chrome_data_dir")]
    pub chrome_data_dir: Option<PathBuf>,
    #[serde(default = "default_clear_cache")]
    pub clear_cache_on_error: bool,
}

fn default_headless() -> bool {
    true
}

fn default_sandbox() -> bool {
    true
}

fn default_chrome_cache_dir() -> Option<PathBuf> {
    // Create unique directory per Chrome instance
    let unique_id = format!(
        "chrome-cache-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    );
    Some(PathBuf::from("/tmp").join(unique_id))
}

fn default_chrome_user_data_dir() -> Option<PathBuf> {
    // Create unique directory per Chrome instance
    let unique_id = format!(
        "chrome-user-data-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    );
    Some(PathBuf::from("/tmp").join(unique_id))
}

fn default_chrome_data_dir() -> Option<PathBuf> {
    // Create unique directory per Chrome instance
    let unique_id = format!(
        "chrome-data-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    );
    Some(PathBuf::from("/tmp").join(unique_id))
}

fn default_clear_cache() -> bool {
    true
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("RENDERER")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
            .and_then(|config: Config| {
                // Create all Chrome directories if they don't exist
                let chrome_dirs = [
                    (&config.chrome_cache_dir, "chrome cache"),
                    (&config.chrome_user_data_dir, "chrome user data"),
                    (&config.chrome_data_dir, "chrome data"),
                ];

                for (dir_option, dir_name) in chrome_dirs.iter() {
                    if let Some(dir_path) = dir_option.as_ref() {
                        if !dir_path.exists() {
                            std::fs::create_dir_all(dir_path).map_err(|e| {
                                ConfigError::Message(format!(
                                    "Failed to create {dir_name} directory: {e}"
                                ))
                            })?;
                        }
                    }
                }

                Ok(config)
            })
    }

    /// Setup Chrome directories with proper permissions for containerized environments
    pub fn setup_chrome_directories(&self) -> Result<(), ConfigError> {
        // Create and set permissions for Chrome temporary directories from config
        let chrome_dirs = [
            (&self.chrome_user_data_dir, "chrome user data"),
            (&self.chrome_data_dir, "chrome data"),
            (&self.chrome_cache_dir, "chrome cache"),
        ];

        for (dir_option, dir_name) in &chrome_dirs {
            if let Some(dir_path) = dir_option.as_ref() {
                // Remove existing directory to prevent permission conflicts
                if dir_path.exists() {
                    fs::remove_dir_all(dir_path).map_err(|e| {
                        ConfigError::Message(format!(
                            "Failed to remove {dir_name} directory {dir_path:?}: {e}"
                        ))
                    })?;
                }

                // Create directory with proper permissions
                fs::create_dir_all(dir_path).map_err(|e| {
                    ConfigError::Message(format!(
                        "Failed to create {dir_name} directory {dir_path:?}: {e}"
                    ))
                })?;

                // Set directory permissions to be writable
                let metadata = fs::metadata(dir_path).map_err(|e| {
                    ConfigError::Message(format!(
                        "Failed to get metadata for {dir_name} directory {dir_path:?}: {e}"
                    ))
                })?;
                let mut permissions = metadata.permissions();
                permissions.set_mode(0o755);
                fs::set_permissions(dir_path, permissions).map_err(|e| {
                    ConfigError::Message(format!(
                        "Failed to set permissions for {dir_name} directory {dir_path:?}: {e}"
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// Get Chrome arguments with paths from config
    pub fn get_chrome_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Add user data directory from config
        if let Some(user_data_dir) = &self.chrome_user_data_dir {
            args.push(format!("--user-data-dir={}", user_data_dir.display()));
        }

        // Add data path from config
        if let Some(data_dir) = &self.chrome_data_dir {
            args.push(format!("--data-path={}", data_dir.display()));
        }

        // Add disk cache directory from config
        if let Some(cache_dir) = &self.chrome_cache_dir {
            args.push(format!("--disk-cache-dir={}", cache_dir.display()));
        }

        args
    }
}
