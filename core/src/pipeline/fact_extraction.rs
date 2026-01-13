use crate::models::document::{Document, DocumentType};
use crate::models::fact::{FactType, NewFact, SourceCitation, BoundingBox};
use crate::models::output::OCRResult;
use regex::Regex;
use serde_json::json;
use std::error::Error;
use uuid::Uuid;

/// Classify document type based on OCR results and file name
pub fn classify_document_type(document: &Document, ocr_results: &[OCRResult]) -> DocumentType {
    let file_name = document.file_name.to_lowercase();
    let ocr_text: String = ocr_results
        .iter()
        .map(|r| r.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    // Check filename first
    if file_name.contains("rent") && file_name.contains("roll") {
        return DocumentType::RentRoll;
    }
    if file_name.contains("p&l") || file_name.contains("profit") || file_name.contains("loss") {
        return DocumentType::ProfitAndLoss;
    }
    if file_name.contains("mortgage") || file_name.contains("loan") {
        return DocumentType::MortgageStatement;
    }
    if file_name.contains("tax") || file_name.contains("1099") || file_name.contains("1098") {
        return DocumentType::TaxDocument;
    }
    if file_name.contains("bank") || file_name.contains("statement") {
        return DocumentType::BankStatement;
    }

    // Check OCR text for keywords
    if ocr_text.contains("rent roll") || (ocr_text.contains("unit") && ocr_text.contains("tenant")) {
        return DocumentType::RentRoll;
    }
    if ocr_text.contains("net operating income") || ocr_text.contains("noi") {
        return DocumentType::ProfitAndLoss;
    }
    if ocr_text.contains("mortgage") || ocr_text.contains("principal") && ocr_text.contains("interest") {
        return DocumentType::MortgageStatement;
    }
    if ocr_text.contains("tax") || ocr_text.contains("form 1099") || ocr_text.contains("form 1098") {
        return DocumentType::TaxDocument;
    }

    DocumentType::Other
}

/// Extract facts from a document based on its type
pub async fn extract_facts_from_document(
    document: &Document,
    ocr_results: &[Vec<OCRResult>],
) -> Result<Vec<NewFact>, Box<dyn Error + Send + Sync>> {
    let document_type = classify_document_type(document, &ocr_results.concat());
    
    match document_type {
        DocumentType::RentRoll => extract_rent_roll_facts(document, ocr_results),
        DocumentType::ProfitAndLoss => extract_pl_facts(document, ocr_results),
        DocumentType::MortgageStatement => extract_mortgage_facts(document, ocr_results),
        DocumentType::TaxDocument => extract_tax_facts(document, ocr_results),
        _ => Ok(vec![]),
    }
}

/// Extract facts from rent roll document
fn extract_rent_roll_facts(
    document: &Document,
    ocr_results: &[Vec<OCRResult>],
) -> Result<Vec<NewFact>, Box<dyn Error + Send + Sync>> {
    let mut facts = Vec::new();
    
    // Extract unit count
    if let Some(fact) = extract_unit_count(document, ocr_results) {
        facts.push(fact);
    }
    
    // Extract occupancy rate
    if let Some(fact) = extract_occupancy_rate(document, ocr_results) {
        facts.push(fact);
    }
    
    // Extract gross scheduled rent
    if let Some(fact) = extract_gross_scheduled_rent(document, ocr_results) {
        facts.push(fact);
    }
    
    // Extract collected rent
    if let Some(fact) = extract_collected_rent(document, ocr_results) {
        facts.push(fact);
    }
    
    Ok(facts)
}

/// Extract facts from P&L document
fn extract_pl_facts(
    document: &Document,
    ocr_results: &[Vec<OCRResult>],
) -> Result<Vec<NewFact>, Box<dyn Error + Send + Sync>> {
    let mut facts = Vec::new();
    
    // Extract operating expenses
    if let Some(fact) = extract_operating_expenses(document, ocr_results) {
        facts.push(fact);
    }
    
    // Extract NOI
    if let Some(fact) = extract_noi(document, ocr_results) {
        facts.push(fact);
    }
    
    // Extract collected rent (may be in P&L as "rental income")
    if let Some(fact) = extract_rental_income(document, ocr_results) {
        facts.push(fact);
    }
    
    Ok(facts)
}

/// Extract facts from mortgage statement
fn extract_mortgage_facts(
    document: &Document,
    ocr_results: &[Vec<OCRResult>],
) -> Result<Vec<NewFact>, Box<dyn Error + Send + Sync>> {
    let mut facts = Vec::new();
    
    // Extract mortgage balance
    if let Some(fact) = extract_mortgage_balance(document, ocr_results) {
        facts.push(fact);
    }
    
    // Extract interest rate
    if let Some(fact) = extract_interest_rate(document, ocr_results) {
        facts.push(fact);
    }
    
    // Extract debt service (monthly payment)
    if let Some(fact) = extract_debt_service(document, ocr_results) {
        facts.push(fact);
    }
    
    Ok(facts)
}

/// Extract facts from tax document
fn extract_tax_facts(
    document: &Document,
    ocr_results: &[Vec<OCRResult>],
) -> Result<Vec<NewFact>, Box<dyn Error + Send + Sync>> {
    let mut facts = Vec::new();
    
    // Extract property value (from assessment)
    if let Some(fact) = extract_property_value(document, ocr_results) {
        facts.push(fact);
    }
    
    Ok(facts)
}

// Helper functions for specific fact extraction

fn extract_unit_count(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let unit_pattern = Regex::new(r"(?i)(total\s+units?|unit\s+count)[:\s]+(\d+)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = unit_pattern.captures(&page_text) {
            if let Some(count_str) = captures.get(2) {
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::UnitCount.as_str().to_string(),
                    label: "Unit Count".to_string(),
                    value: count_str.as_str().to_string(),
                    unit: Some("units".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.8),
                });
            }
        }
    }
    
    None
}

fn extract_occupancy_rate(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let occupancy_pattern = Regex::new(r"(?i)occupancy[:\s]+(\d+\.?\d*)%").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = occupancy_pattern.captures(&page_text) {
            if let Some(rate_str) = captures.get(1) {
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::OccupancyRate.as_str().to_string(),
                    label: "Occupancy Rate".to_string(),
                    value: rate_str.as_str().to_string(),
                    unit: Some("%".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.85),
                });
            }
        }
    }
    
    None
}

fn extract_gross_scheduled_rent(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let rent_pattern = Regex::new(r"(?i)gross\s+scheduled\s+rent[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = rent_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(1) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::GrossScheduledRent.as_str().to_string(),
                    label: "Gross Scheduled Rent".to_string(),
                    value: amount,
                    unit: Some("USD/year".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.9),
                });
            }
        }
    }
    
    None
}

fn extract_collected_rent(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let rent_pattern = Regex::new(r"(?i)collected\s+rent[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = rent_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(1) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::CollectedRent.as_str().to_string(),
                    label: "Collected Rent".to_string(),
                    value: amount,
                    unit: Some("USD/year".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.9),
                });
            }
        }
    }
    
    None
}

fn extract_operating_expenses(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let expense_pattern = Regex::new(r"(?i)(operating\s+expenses?|total\s+expenses?)[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = expense_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(2) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::OperatingExpenses.as_str().to_string(),
                    label: "Operating Expenses".to_string(),
                    value: amount,
                    unit: Some("USD/year".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.85),
                });
            }
        }
    }
    
    None
}

fn extract_noi(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let noi_pattern = Regex::new(r"(?i)(net\s+operating\s+income|noi)[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = noi_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(2) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::NetOperatingIncome.as_str().to_string(),
                    label: "Net Operating Income".to_string(),
                    value: amount,
                    unit: Some("USD/year".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.95),
                });
            }
        }
    }
    
    None
}

fn extract_rental_income(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let income_pattern = Regex::new(r"(?i)rental\s+income[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = income_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(1) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::CollectedRent.as_str().to_string(),
                    label: "Collected Rent".to_string(),
                    value: amount,
                    unit: Some("USD/year".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.85),
                });
            }
        }
    }
    
    None
}

fn extract_mortgage_balance(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let balance_pattern = Regex::new(r"(?i)(principal\s+balance|outstanding\s+balance|loan\s+balance)[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = balance_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(2) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::MortgageBalance.as_str().to_string(),
                    label: "Mortgage Balance".to_string(),
                    value: amount,
                    unit: Some("USD".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.9),
                });
            }
        }
    }
    
    None
}

fn extract_interest_rate(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let rate_pattern = Regex::new(r"(?i)interest\s+rate[:\s]+(\d+\.?\d*)%").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = rate_pattern.captures(&page_text) {
            if let Some(rate_str) = captures.get(1) {
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::InterestRate.as_str().to_string(),
                    label: "Interest Rate".to_string(),
                    value: rate_str.as_str().to_string(),
                    unit: Some("%".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.95),
                });
            }
        }
    }
    
    None
}

fn extract_debt_service(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let payment_pattern = Regex::new(r"(?i)(monthly\s+payment|debt\s+service)[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = payment_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(2) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::DebtService.as_str().to_string(),
                    label: "Debt Service".to_string(),
                    value: amount,
                    unit: Some("USD/month".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.9),
                });
            }
        }
    }
    
    None
}

fn extract_property_value(document: &Document, ocr_results: &[Vec<OCRResult>]) -> Option<NewFact> {
    let value_pattern = Regex::new(r"(?i)(assessed\s+value|property\s+value|market\s+value)[:\s]+\$?([\d,]+\.?\d*)").ok()?;
    
    for (page_idx, page_results) in ocr_results.iter().enumerate() {
        let page_text = page_results.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join(" ");
        
        if let Some(captures) = value_pattern.captures(&page_text) {
            if let Some(amount_str) = captures.get(2) {
                let amount = amount_str.as_str().replace(",", "");
                
                let citation = SourceCitation {
                    document: document.file_name.clone(),
                    page: (page_idx + 1) as i32,
                    line: Some(captures.get(0)?.as_str().to_string()),
                    bbox: None,
                };
                
                return Some(NewFact {
                    fact_id: Uuid::new_v4().to_string(),
                    document_id: document.document_id.clone(),
                    deal_id: document.deal_id.clone(),
                    fact_type: FactType::PropertyValue.as_str().to_string(),
                    label: "Property Value".to_string(),
                    value: amount,
                    unit: Some("USD".to_string()),
                    source_citation: json!(citation),
                    status: "pending_approval".to_string(),
                    confidence_score: Some(0.8),
                });
            }
        }
    }
    
    None
}

