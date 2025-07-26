use calamine::{open_workbook_auto, Reader, SheetVisible, Sheets};
use std::fs::File;
use std::io::BufReader;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use thiserror::Error;

/// Represents worksheet information: (name, start_position, end_position, visible_status)
pub type WorksheetInfo = (String, Option<(u32, u32)>, Option<(u32, u32)>, SheetVisible);

#[derive(Debug, Error)]
pub enum ExcelError {
    #[error("Excel parsing error: {0}")]
    CalamineError(calamine::Error),
    #[error("Invalid Excel file: {0}")]
    InvalidFile(String),
    #[error("No worksheets found in Excel file")]
    NoSheetsFound,
    #[error("IO error reading Excel file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Excel file caused internal error: {0}")]
    PanicError(String),
}

impl From<calamine::Error> for ExcelError {
    fn from(error: calamine::Error) -> Self {
        match &error {
            calamine::Error::Msg(msg) if msg.contains("Cannot detect file format") => {
                ExcelError::InvalidFile(msg.to_string())
            }
            _ => ExcelError::CalamineError(error),
        }
    }
}

pub fn open_workbook(file_path: &Path) -> Result<Sheets<BufReader<File>>, ExcelError> {
    if !file_path.exists() {
        return Err(ExcelError::InvalidFile(
            file_path.to_string_lossy().to_string(),
        ));
    }

    // Use panic::catch_unwind to safely handle potential panics in calamine
    let result = panic::catch_unwind(|| open_workbook_auto(file_path));

    match result {
        Ok(Ok(workbook)) => Ok(workbook),
        Ok(Err(e)) => Err(ExcelError::from(e)),
        Err(e) => {
            let error_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic occurred while opening Excel file".to_string()
            };
            Err(ExcelError::PanicError(error_msg))
        }
    }
}

pub fn count_sheets(file_path: &Path, count_hidden: bool) -> Result<u32, ExcelError> {
    let mut workbook = open_workbook(file_path)?;

    // Use panic::catch_unwind to safely handle potential panics in calamine
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let worksheets = workbook.worksheets();
        let sheets_metadata = workbook.sheets_metadata();

        let count = if count_hidden {
            // If count_hidden is true, count all sheets
            worksheets.len()
        } else {
            // If count_hidden is false, only count visible sheets
            worksheets
                .iter()
                .filter(|(name, _)| {
                    sheets_metadata
                        .iter()
                        .find(|sheet| sheet.name == *name)
                        .map(|sheet| sheet.visible == SheetVisible::Visible)
                        .unwrap_or(true) // Default to visible if not found
                })
                .count()
        };

        count
    }));

    match result {
        Ok(sheet_count) => {
            if sheet_count == 0 {
                Err(ExcelError::NoSheetsFound)
            } else {
                Ok(sheet_count as u32)
            }
        }
        Err(_) => Err(ExcelError::PanicError(
            "Calamine library encountered an internal error while counting sheets. The file may be corrupted or contain invalid data.".to_string()
        )),
    }
}

/// Safely get worksheet information without panicking
pub fn get_worksheets_info(file_path: &Path) -> Result<Vec<WorksheetInfo>, ExcelError> {
    let mut workbook = open_workbook(file_path)?;

    // Use panic::catch_unwind to safely handle potential panics in calamine
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let worksheets = workbook.worksheets();
        let sheets_metadata = workbook.sheets_metadata();
        let mut sheet_infos = Vec::new();

        for (name, range) in worksheets {
            // Safely get range information
            let (start_pos, end_pos) = match panic::catch_unwind(AssertUnwindSafe(|| {
                let start = range.start();
                let end = range.end();
                (start, end)
            })) {
                Ok((start, end)) => (start, end),
                Err(_) => {
                    // If we can't get range info safely, error out
                    return Err(ExcelError::PanicError(format!(
                        "Could not safely read range for sheet '{name}'"
                    )));
                }
            };

            // Get visibility status from sheet metadata
            let visible_status = sheets_metadata
                .iter()
                .find(|sheet| sheet.name == name)
                .map(|sheet| sheet.visible)
                .unwrap_or(SheetVisible::Visible); // Default to visible if not found

            sheet_infos.push((name, start_pos, end_pos, visible_status));
        }

        Ok(sheet_infos)
    }));

    match result {
        Ok(Ok(sheet_infos)) => {
            if sheet_infos.is_empty() {
                Err(ExcelError::NoSheetsFound)
            } else {
                Ok(sheet_infos.clone())
            }
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err(ExcelError::PanicError(
            "Calamine library encountered an internal error while reading worksheet information. The file may be corrupted or contain invalid data.".to_string()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_count_sheets() {
        let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        file_path.push("/home/akhilesh/Downloads/1_runtime.xls");

        match count_sheets(&file_path, false) {
            Ok(result) => assert_eq!(result, 1),
            Err(ExcelError::InvalidFile(_)) => {
                // File doesn't exist, which is expected in CI/different environments
                println!("Test file not found, skipping assertion");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[tokio::test]
    async fn test_count_sheets_invalid_file() {
        let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        file_path.push("/nonexistent/file.xlsx");
        let result = count_sheets(&file_path, false);
        assert!(matches!(result, Err(ExcelError::InvalidFile(_))));
    }

    #[tokio::test]
    async fn test_excel_error_display() {
        let error = ExcelError::NoSheetsFound;
        assert_eq!(error.to_string(), "No worksheets found in Excel file");

        let error = ExcelError::InvalidFile("test.xlsx".to_string());
        assert_eq!(error.to_string(), "Invalid Excel file: test.xlsx");

        let error = ExcelError::PanicError("test panic".to_string());
        assert_eq!(
            error.to_string(),
            "Excel file caused internal error: test panic"
        );
    }
}
