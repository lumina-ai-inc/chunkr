use crate::configs::renderer_config::Config;
use crate::models::output::BoundingBox;
use headless_chrome::protocol::cdp::Page;
use headless_chrome::{Browser, LaunchOptions};
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::Arc;
use tempfile::Builder;
use tempfile::NamedTempFile;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Chrome error: {0}")]
    Chrome(String),
    #[error("Config error: {0}")]
    Config(String),
    #[error("Chrome cache cleanup error: {0}")]
    CacheCleanup(String),
}

#[derive(Serialize, Deserialize, Debug)]
struct ElementData {
    html: String,
    rect: RectData,
}

#[derive(Serialize, Deserialize, Debug)]
struct RectData {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
    width: f64,
    height: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HtmlRect {
    pub top: f64,
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    pub id: String,
    pub bbox: BoundingBox,
    pub html: String,
    pub rect: HtmlRect,
}

impl Element {
    /// Create a new Element with BoundingBox calculated from HtmlRect and DPI
    pub fn new(html: String, rect: HtmlRect, dpi: f64) -> Self {
        // Convert from screen coordinates to document coordinates using DPI
        // Standard web DPI is 96, so we normalize to that
        let scale_factor = 96.0 / dpi;

        let bbox = BoundingBox::new(
            (rect.left * scale_factor) as f32,
            (rect.top * scale_factor) as f32,
            (rect.width * scale_factor) as f32,
            (rect.height * scale_factor) as f32,
        );

        Self {
            id: default_id(),
            bbox,
            html,
            rect,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Capture {
    pub id: String,
    pub image: Arc<NamedTempFile>,
    pub width: f64,
    pub height: f64,
    pub dpi: f64,
    pub elements: Vec<Element>,
}

impl Capture {
    pub fn new(
        image: NamedTempFile,
        width: f64,
        height: f64,
        dpi: f64,
        elements: Vec<Element>,
    ) -> Self {
        Self {
            id: default_id(),
            image: Arc::new(image),
            width,
            height,
            dpi,
            elements,
        }
    }
}

fn default_id() -> String {
    Uuid::new_v4().to_string()
}

/// HTML to Image renderer using headless Chrome.
/// Provides exact sizing with no wasted whitespace.
pub struct HTMLToImageRenderer {
    browser: Browser,
}

impl HTMLToImageRenderer {
    /// Initialize the renderer with a headless Chrome browser instance.
    /// Includes retry logic and cache cleanup for corrupted Chrome downloads.
    pub fn new() -> Result<Self, RendererError> {
        let config = Config::from_env().map_err(|e| RendererError::Config(e.to_string()))?;

        // Try to create browser with retry and cleanup logic
        let browser = Self::create_browser_with_retry(&config)?;

        Ok(Self { browser })
    }

    fn create_browser_with_retry(config: &Config) -> Result<Browser, RendererError> {
        const MAX_RETRIES: u32 = 3;
        let mut last_error = None;

        for attempt in 1..=MAX_RETRIES {
            match Self::try_create_browser(config) {
                Ok(browser) => {
                    return Ok(browser);
                }
                Err(e) => {
                    println!("❌ Chrome browser creation failed (attempt {attempt}): {e}");
                    last_error = Some(e);

                    // Check if this is a zip corruption error and attempt cleanup
                    if let RendererError::Chrome(ref error_msg) = last_error.as_ref().unwrap() {
                        if error_msg.contains("invalid Zip archive") || error_msg.contains("EOCD") {
                            println!(
                                "🧹 Detected corrupted Chrome download, attempting cache cleanup",
                            );
                            if let Err(cleanup_err) = Self::cleanup_chrome_cache(config) {
                                println!("⚠️ Cache cleanup failed: {cleanup_err}");
                            }
                        }
                    }

                    // For permission errors, try cleaning up Chrome directories and retry
                    if let RendererError::Chrome(ref error_msg) = last_error.as_ref().unwrap() {
                        if error_msg.contains("Permission denied") {
                            println!(
                                "🧹 Permission error detected, cleaning up Chrome directories"
                            );
                            if let Err(setup_err) = config.setup_chrome_directories() {
                                println!("⚠️ Chrome directory setup failed: {setup_err}");
                            }
                        }
                    }

                    // Wait before retry (exponential backoff)
                    if attempt < MAX_RETRIES {
                        let wait_time = std::time::Duration::from_secs(2_u64.pow(attempt - 1));
                        std::thread::sleep(wait_time);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            RendererError::Chrome("Failed to create browser after all retries".to_string())
        }))
    }

    fn try_create_browser(config: &Config) -> Result<Browser, RendererError> {
        // Check Chrome executable permissions before attempting launch
        let chrome_executable = config
            .chrome_path
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("/usr/bin/google-chrome"));

        if !chrome_executable.exists() {
            println!("⚠️ Chrome executable not found at: {chrome_executable:?}");
        }

        // Build launch options
        let mut options_builder = LaunchOptions::default_builder();
        options_builder
            .headless(config.headless)
            .sandbox(config.sandbox);

        // Ensure Chrome temporary directories exist and have proper permissions
        config
            .setup_chrome_directories()
            .map_err(|e| RendererError::Config(e.to_string()))?;

        // Use custom Chrome path if provided
        if let Some(chrome_path) = &config.chrome_path {
            if chrome_path.exists() {
                options_builder.path(Some(chrome_path.clone()));
            } else {
                println!("⚠️ Custom Chrome path does not exist: {chrome_path:?}");
            }
        }

        // Set user data directory for Chrome cache
        if let Some(cache_dir) = &config.chrome_cache_dir {
            options_builder.user_data_dir(Some(cache_dir.clone()));
        }

        // Add Chrome arguments for better containerized execution
        let mut chrome_args = vec![
            "--no-first-run".to_string(),
            "--no-default-browser-check".to_string(),
            "--disable-background-timer-throttling".to_string(),
            "--disable-backgrounding-occluded-windows".to_string(),
            "--disable-renderer-backgrounding".to_string(),
            "--disable-features=TranslateUI".to_string(),
            "--disable-ipc-flooding-protection".to_string(),
            "--disable-dev-shm-usage".to_string(),
            "--disable-gpu".to_string(),
            "--disable-software-rasterizer".to_string(),
            "--disable-extensions".to_string(),
            "--disable-plugins".to_string(),
            "--disable-web-security".to_string(),
            "--allow-running-insecure-content".to_string(),
            "--ignore-certificate-errors".to_string(),
            "--ignore-ssl-errors".to_string(),
            "--ignore-certificate-errors-spki-list".to_string(),
            "--ignore-certificate-errors-ssl-errors".to_string(),
            "--memory-pressure-off".to_string(),
            "--max_old_space_size=4096".to_string(),
            // Additional flags for container compatibility and permission issues
            "--disable-background-networking".to_string(),
            "--disable-default-apps".to_string(),
            "--disable-sync".to_string(),
            "--disable-translate".to_string(),
            "--hide-scrollbars".to_string(),
            "--metrics-recording-only".to_string(),
            "--mute-audio".to_string(),
            "--no-pings".to_string(),
            "--disable-logging".to_string(),
            "--disable-gpu-logging".to_string(),
            "--disable-crash-reporter".to_string(),
            "--disable-in-process-stack-traces".to_string(),
            "--disable-build-revision-check".to_string(),
            "--disable-device-discovery-notifications".to_string(),
            "--use-mock-keychain".to_string(),
            "--test-type".to_string(),
            "--single-process".to_string(), // Run everything in single process to avoid permission issues
            // Additional containerization and permission flags
            "--disable-zygote".to_string(), // Disable zygote process for containers
            "--disable-features=VizDisplayCompositor".to_string(), // Disable compositor for container compatibility
            "--homedir=/tmp".to_string(),                          // Set home dir to tmp
            "--disable-features=AudioServiceOutOfProcess".to_string(), // Disable audio service process
            "--disable-features=MediaRouter".to_string(),              // Disable media router
            "--disable-namespace-sandbox".to_string(), // Disable namespace sandbox for containers
            "--disable-breakpad".to_string(),          // Disable crash reporter
            "--disable-component-extensions-with-background-pages".to_string(),
            "--disable-component-cloud-policy".to_string(),
            "--disable-domain-reliability".to_string(),
            "--disable-client-side-phishing-detection".to_string(),
            "--disable-hang-monitor".to_string(),
            "--disable-prompt-on-repost".to_string(),
            "--disable-session-crashed-bubble".to_string(),
            "--disable-speech-api".to_string(),
            "--disable-web-resources".to_string(),
            "--disable-webgl".to_string(),
            "--disable-webrtc".to_string(),
            "--disable-print-preview".to_string(),
            "--disable-permissions-api".to_string(),
            "--disable-notifications".to_string(),
            "--disable-gesture-typing".to_string(),
            "--disable-file-system".to_string(),
            "--disable-databases".to_string(),
            "--disable-application-cache".to_string(),
            "--disable-accelerated-2d-canvas".to_string(),
            "--disable-accelerated-video-decode".to_string(),
            "--disable-accelerated-video-encode".to_string(),
            "--disable-blink-features=AutomationControlled".to_string(), // Prevent automation detection
        ];

        // Add directory-specific Chrome arguments from config
        chrome_args.extend(config.get_chrome_args());

        // Add sandbox-specific arguments
        if !config.sandbox {
            chrome_args.extend(vec![
                "--no-sandbox".to_string(),
                "--disable-setuid-sandbox".to_string(),
            ]);
        }

        options_builder.args(
            chrome_args
                .iter()
                .map(|s| s.as_ref())
                .collect::<Vec<&std::ffi::OsStr>>(),
        );

        let options = options_builder
            .build()
            .map_err(|e| RendererError::Chrome(format!("Launch options error: {e}")))?;

        Browser::new(options)
            .map_err(|e| RendererError::Chrome(format!("Browser launch error: {e}")))
    }

    fn cleanup_chrome_cache(config: &Config) -> Result<(), RendererError> {
        if !config.clear_cache_on_error {
            return Ok(());
        }

        // Clean up potential Chrome cache directories
        let cache_dirs = vec![
            config.chrome_cache_dir.clone(),
            Some(std::env::temp_dir().join("rust_headless_chrome")),
            Some(
                std::env::current_dir()
                    .unwrap_or_default()
                    .join("chrome-cache"),
            ),
            Some(std::env::current_dir().unwrap_or_default().join(".cache")),
        ];

        for cache_dir in cache_dirs.into_iter().flatten() {
            if cache_dir.exists() && cache_dir.is_dir() {
                match fs::remove_dir_all(&cache_dir) {
                    Ok(()) => {}
                    Err(e) => {
                        println!("⚠️ Failed to clean cache directory {cache_dir:?}: {e}")
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_html_suffix(&self, html_file: &NamedTempFile) -> Result<(), RendererError> {
        let path = html_file.path();
        if path.extension().is_none_or(|ext| ext != "html") {
            return Err(
                RendererError::Chrome(
                    "Input file must have .html extension. Use Builder::new().suffix(\".html\").tempfile() when creating the NamedTempFile.".to_string()
                )
            );
        }
        Ok(())
    }

    /// Helper function to set up browser tab and navigate to HTML file.
    /// Returns the configured tab ready for rendering.
    fn setup_html_tab(
        &self,
        html_file: &NamedTempFile,
    ) -> Result<std::sync::Arc<headless_chrome::Tab>, RendererError> {
        self.validate_html_suffix(html_file)?;

        // Create new tab (reusing existing browser instance)
        let tab = self
            .browser
            .new_tab()
            .map_err(|e| RendererError::Chrome(format!("New tab error: {e}")))?;

        // Navigate to file URL directly
        let file_url = format!("file://{}", html_file.path().to_string_lossy());
        tab.navigate_to(&file_url)
            .map_err(|e| RendererError::Chrome(format!("Navigate error: {e}")))?;

        // Wait for content to load
        tab.wait_for_element("body")
            .map_err(|e| RendererError::Chrome(format!("Wait for element error: {e}")))?;

        Ok(tab)
    }

    /// Helper function to get content dimensions from the loaded HTML page.
    /// Returns (width, height) in pixels.
    fn get_content_dimensions(
        &self,
        tab: &std::sync::Arc<headless_chrome::Tab>,
    ) -> Result<(f64, f64), RendererError> {
        // Get the actual content dimensions
        let content_size = tab
            .evaluate(
                r#"
            (() => {
                const body = document.body;
                const html = document.documentElement;
                
                // Get table dimensions if present to handle wide tables
                const table = document.querySelector('table');
                let tableWidth = 0;
                if (table) {
                    tableWidth = table.scrollWidth || table.offsetWidth || 0;
                }
                
                return {
                    width: Math.max(
                        body.scrollWidth, body.offsetWidth, 
                        html.clientWidth, html.scrollWidth, html.offsetWidth,
                        tableWidth
                    ),
                    height: Math.max(
                        body.scrollHeight, body.offsetHeight,
                        html.clientHeight, html.scrollHeight, html.offsetHeight
                    )
                };
            })()
        "#,
                false,
            )
            .map_err(|e| RendererError::Chrome(format!("Evaluate error: {e}")))?;

        // Extract width and height from the result
        let (width, height) = if let Some(preview) = content_size.preview.as_ref() {
            let width_prop = preview
                .properties
                .iter()
                .find(|prop| prop.name == "width")
                .and_then(|prop| prop.value.as_ref().and_then(|v| v.parse::<f64>().ok()))
                .ok_or(RendererError::Chrome(
                    "Width not found in preview".to_string(),
                ))?;

            let height_prop = preview
                .properties
                .iter()
                .find(|prop| prop.name == "height")
                .and_then(|prop| prop.value.as_ref().and_then(|v| v.parse::<f64>().ok()))
                .ok_or(RendererError::Chrome(
                    "Height not found in preview".to_string(),
                ))?;

            (width_prop, height_prop)
        } else {
            return Err(RendererError::Chrome(
                "No width/height data found in response".to_string(),
            ));
        };

        Ok((width, height))
    }

    /// Helper function to set up viewport sizing for exact content capture (used for images).
    fn setup_viewport_sizing(
        &self,
        tab: &std::sync::Arc<headless_chrome::Tab>,
    ) -> Result<(), RendererError> {
        let (width, height) = self.get_content_dimensions(tab)?;

        // Set the viewport size to match content dimensions
        tab.call_method(
            headless_chrome::protocol::cdp::Emulation::SetDeviceMetricsOverride {
                width: width as u32,
                height: height as u32,
                device_scale_factor: 1.0,
                mobile: false,
                scale: Some(1.0),
                screen_width: Some(width as u32),
                screen_height: Some(height as u32),
                position_x: Some(0),
                position_y: Some(0),
                dont_set_visible_size: Some(false),
                screen_orientation: None,
                viewport: None,
                device_posture: None,
                display_feature: None,
            },
        )
        .map_err(|e| RendererError::Chrome(format!("Set viewport error: {e}")))?;

        Ok(())
    }

    /// Extract HTML elements from the DOM with their bounding rectangles
    fn extract_html_elements(
        &self,
        tab: &std::sync::Arc<headless_chrome::Tab>,
        dpi: f64,
    ) -> Result<Vec<Element>, RendererError> {
        let elements_js = r#"
            (() => {
                const elements = [];
                
                // Get all visible elements with meaningful content
                const allElements = document.querySelectorAll('*');
                
                for (const element of allElements) {
                    // Skip elements that are not visible or have no content
                    if (element.offsetWidth === 0 || element.offsetHeight === 0) {
                        continue;
                    }
                    
                    // Skip script, style, and meta elements
                    if (['SCRIPT', 'STYLE', 'META', 'LINK', 'TITLE', 'HEAD'].includes(element.tagName)) {
                        continue;
                    }
                    
                    // Get bounding rectangle
                    const rect = element.getBoundingClientRect();
                    
                    // Skip elements with no dimensions
                    if (rect.width === 0 || rect.height === 0) {
                        continue;
                    }
                    
                    // Get HTML content
                    let html = element.outerHTML;
                    
                    elements.push({
                        html: html,
                        rect: {
                            left: rect.left,
                            top: rect.top,
                            right: rect.right,
                            bottom: rect.bottom,
                            width: rect.width,
                            height: rect.height
                        }
                    });
                }
                
                return JSON.stringify(elements);
            })()
        "#;

        let result = tab
            .evaluate(elements_js, false)
            .map_err(|e| RendererError::Chrome(format!("Extract elements error: {e}")))?;

        // Parse the JSON result
        let elements_json = if let Some(value) = result.value.as_ref() {
            match value {
                serde_json::Value::String(s) => s.clone(),
                _ => {
                    return Err(RendererError::Chrome(
                        "Expected string result from element extraction".to_string(),
                    ));
                }
            }
        } else {
            return Err(RendererError::Chrome(
                "No result from element extraction".to_string(),
            ));
        };

        // Parse elements from JSON using proper structs
        let elements_data: Vec<ElementData> = serde_json::from_str(&elements_json)
            .map_err(|e| RendererError::Chrome(format!("JSON parse error: {e}")))?;

        let mut elements = Vec::new();

        for element_data in elements_data {
            let rect = HtmlRect {
                left: element_data.rect.left,
                top: element_data.rect.top,
                right: element_data.rect.right,
                bottom: element_data.rect.bottom,
                width: element_data.rect.width,
                height: element_data.rect.height,
            };

            if rect.width > 0.0 && rect.height > 0.0 {
                elements.push(Element::new(element_data.html, rect, dpi));
            }
        }

        Ok(elements)
    }

    /// Render HTML content to an image with exact content sizing.
    ///
    /// # Arguments
    /// * `html_file` - NamedTempFile containing HTML content (must have .html extension)
    ///
    /// # Returns
    /// Capture struct containing the generated image and metadata
    pub fn render_html(&self, html_file: &NamedTempFile) -> Result<Capture, RendererError> {
        let tab = self.setup_html_tab(html_file)?;
        self.setup_viewport_sizing(&tab)?;
        let output_file = Builder::new().suffix(".jpg").tempfile()?;

        self.scroll_and_wait_for_images(&tab)?;

        // Get content dimensions for the capture metadata
        let (width, height) = self.get_content_dimensions(&tab)?;
        let dpi = 96.0; // Standard web DPI

        // Extract HTML elements before taking screenshot
        let elements = self.extract_html_elements(&tab, dpi)?;

        // Take screenshot with JPEG format - no clip needed since viewport is exact size
        let screenshot_data = tab
            .capture_screenshot(
                Page::CaptureScreenshotFormatOption::Jpeg,
                None,
                None, // No clip needed since viewport is exact size
                true, // from_surface
            )
            .map_err(|e| RendererError::Chrome(format!("Screenshot error: {e}")))?;

        // Save screenshot to file
        fs::write(output_file.path(), screenshot_data)?;

        // Create and return Capture struct
        let capture = Capture::new(output_file, width, height, dpi, elements);

        Ok(capture)
    }

    /// Scroll to bottom and wait for all images to load
    fn scroll_and_wait_for_images(
        &self,
        tab: &std::sync::Arc<headless_chrome::Tab>,
    ) -> Result<(), RendererError> {
        // Scroll to bottom of page to trigger lazy loading
        tab.evaluate("window.scrollTo(0, document.body.scrollHeight);", false)
            .map_err(|e| RendererError::Chrome(format!("Scroll error: {e}")))?;

        // Wait for all images to load using JavaScript
        let wait_for_images_js = r#"
            new Promise((resolve, reject) => {
                const images = Array.from(document.querySelectorAll('img'));
                
                if (images.length === 0) {
                    resolve(true);
                    return;
                }
                
                let loadedCount = 0;
                const totalImages = images.length;
                
                const checkComplete = () => {
                    loadedCount++;
                    if (loadedCount >= totalImages) {
                        resolve(true);
                    }
                };
                
                images.forEach(img => {
                    if (img.complete) {
                        checkComplete();
                    } else {
                        img.addEventListener('load', checkComplete);
                        img.addEventListener('error', () => reject(new Error('Image loading error')));
                    }
                });
                
                // Timeout after 10 seconds to avoid infinite waiting
                setTimeout(() => {
                    console.warn('Image loading timeout after 10 seconds');
                    reject(new Error('Image loading timeout'));
                }, 10000);
            })
        "#;

        tab.evaluate(wait_for_images_js, true)
            .map_err(|e| RendererError::Chrome(format!("Wait for images error: {e}")))?;

        Ok(())
    }

    /// Render HTML content to a PDF with exact content sizing.
    ///
    /// # Arguments
    /// * `html_file` - NamedTempFile containing HTML content (must have .html extension)
    ///
    /// # Returns
    /// NamedTempFile containing the generated PDF
    pub fn render_html_to_pdf(
        &self,
        html_file: &NamedTempFile,
        dpi: Option<f64>,
    ) -> Result<NamedTempFile, RendererError> {
        let tab = self.setup_html_tab(html_file)?;
        let output_file = Builder::new().suffix(".pdf").tempfile()?;
        let default_dpi = 96.0;
        let dpi = dpi.unwrap_or(default_dpi);
        // Set up viewport sizing for consistent layout
        self.setup_viewport_sizing(&tab)?;

        // Scroll to bottom and wait for images to load
        self.scroll_and_wait_for_images(&tab)?;

        // Get content dimensions and convert to inches for PDF paper size
        let (width_px, height_px) = self.get_content_dimensions(&tab)?;
        let width_inches = width_px / dpi;
        let height_inches = height_px / dpi;

        // Generate PDF with exact content dimensions
        let pdf_options = headless_chrome::types::PrintToPdfOptions {
            paper_width: Some(width_inches),
            paper_height: Some(height_inches),
            print_background: Some(true),
            margin_top: Some(0.0),
            margin_bottom: Some(0.0),
            margin_left: Some(0.0),
            margin_right: Some(0.0),
            prefer_css_page_size: Some(false), // Use our calculated dimensions instead
            generate_tagged_pdf: Some(true),
            ..Default::default()
        };

        let pdf_data = tab
            .print_to_pdf(Some(pdf_options))
            .map_err(|e| RendererError::Chrome(format!("PDF generation error: {e}")))?;

        // Save PDF to file
        fs::write(output_file.path(), pdf_data)?;

        Ok(output_file)
    }
}

// Implement Drop to ensure browser cleanup
impl Drop for HTMLToImageRenderer {
    fn drop(&mut self) {
        // Browser will be automatically cleaned up when dropped
        log::debug!("HTMLToImageRenderer dropped, browser cleaned up");
    }
}

/// Convenience function to render HTML using the global singleton
///
/// # Arguments
/// * `html_file` - NamedTempFile containing HTML content (must have .html extension)
///
/// # Returns
/// NamedTempFile containing the generated image
pub fn render_html_to_image(html_file: &NamedTempFile) -> Result<Capture, RendererError> {
    HTMLToImageRenderer::new()?.render_html(html_file)
}

/// Convenience function to render HTML to PDF using the global singleton
///
/// # Arguments
/// * `html_file` - NamedTempFile containing HTML content (must have .html extension)
/// * `dpi` - DPI of the output PDF (default is 96)
///
/// # Returns
/// NamedTempFile containing the generated PDF
pub fn render_html_to_pdf(
    html_file: &NamedTempFile,
    dpi: Option<f64>,
) -> Result<NamedTempFile, RendererError> {
    HTMLToImageRenderer::new()?.render_html_to_pdf(html_file, dpi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::Builder;

    #[test]
    fn test_basic_html_rendering() {
        let html = r#"
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table { border-collapse: collapse; font-family: Arial, sans-serif; }
                th, td { border: 1px solid #888; padding: 8px; }
                th { background: #E0E0E0; font-weight: bold; }
            </style>
        </head>
        <body>
            <table>
                <tr><th>Name</th><th>Age</th></tr>
                <tr><td>Alice</td><td>30</td></tr>
                <tr><td>Bob</td><td>25</td></tr>
            </table>
        </body>
        </html>
        "#;
        let html_file = Builder::new().suffix(".html").tempfile().unwrap();
        fs::write(html_file.path(), html).unwrap();

        let result = render_html_to_image(&html_file);

        assert!(result.is_ok());
        let capture = result.unwrap();
        assert!(capture.image.path().exists());

        // Save the final output
        let output_dir = PathBuf::from("output/renderer");
        let final_output = output_dir.join("input.png");
        save_output_file(&capture.image, &final_output);
    }

    #[test]
    fn test_render_html_file() {
        let mut input_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_file_path.push("input/sheet.html");

        // Read file content and create NamedTempFile with .html extension
        let content = fs::read_to_string(&input_file_path).unwrap();
        let html_file = Builder::new().suffix(".html").tempfile().unwrap();
        fs::write(html_file.path(), content).unwrap();

        let capture = render_html_to_image(&html_file).unwrap();

        assert!(capture.image.path().exists());

        let output_dir = PathBuf::from("output/excel/renders");
        let final_output = output_dir.join("input.png");
        save_output_file(&capture.image, &final_output);
        let elements_path = output_dir.join("elements.json");
        fs::write(
            elements_path,
            serde_json::to_string(&capture.elements).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn test_render_html_file_to_pdf() {
        let mut input_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_file_path.push("input/test.html");

        // Read file content and create NamedTempFile with .html extension
        let content = fs::read_to_string(&input_file_path).unwrap();
        let html_file = Builder::new().suffix(".html").tempfile().unwrap();
        fs::write(html_file.path(), content).unwrap();

        let output_file = render_html_to_pdf(&html_file, None).unwrap();

        assert!(output_file.path().exists());

        let output_dir = PathBuf::from("output/excel/renders");
        let final_output = output_dir.join("input.pdf");
        save_output_file(&output_file, &final_output);
    }

    fn save_output_file(input_file: &NamedTempFile, output_file: &PathBuf) {
        let output_dir = output_file.parent().unwrap();
        fs::create_dir_all(output_dir).unwrap();
        fs::copy(input_file.path(), output_file).unwrap();
        println!("Output file: {output_file:?}");
    }
}
