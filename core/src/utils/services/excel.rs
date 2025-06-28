use calamine::{open_workbook_auto, Reader};

pub fn count_sheets(file_path: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let mut workbook = open_workbook_auto(file_path)?;
    Ok(workbook.worksheets().len() as u32)
}
