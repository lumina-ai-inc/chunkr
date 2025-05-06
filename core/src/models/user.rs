use chrono::{ DateTime, Utc };
use postgres_types::{ FromSql, ToSql };
use serde::{ Deserialize, Serialize };
use strum_macros::{ Display, EnumString };
use utoipa::ToSchema;
use std::fmt;

#[derive(
    Serialize,
    Deserialize,
    Debug,
    PartialEq,
    Eq,
    Clone,
    Display,
    EnumString,
    FromSql,
    ToSql,
    ToSchema
)]
#[postgres(name = "tier")]
pub enum Tier {
    Free,
    PayAsYouGo,
    Enterprise,
    SelfHosted,
    Dev,
    Starter,
    Team,
}

#[derive(
    Serialize,
    Deserialize,
    Debug,
    PartialEq,
    Eq,
    Clone,
    Display,
    EnumString,
    Hash,
    ToSchema,
    ToSql,
    FromSql
)]
#[postgres(name = "usage_type")]
pub enum UsageType {
    Fast,
    HighQuality,
    Segment,
    Page,
}

impl UsageType {
    pub fn get_unit(&self) -> String {
        match self {
            UsageType::Fast => "Page".to_string(),
            UsageType::HighQuality => "Page".to_string(),
            UsageType::Segment => "Segment".to_string(),
            UsageType::Page => "Page".to_string(),
        }
    }

    pub fn get_usage_limit(&self, tier: &Tier) -> i32 {
        match tier {
            Tier::Free =>
                match self {
                    UsageType::Fast => 1000,
                    UsageType::HighQuality => 500,
                    UsageType::Segment => 250,
                    UsageType::Page => 1000,
                }
            Tier::PayAsYouGo =>
                match self {
                    UsageType::Fast => 1000000,
                    UsageType::HighQuality => 1000000,
                    UsageType::Segment => 1000000,
                    UsageType::Page => 10000000,
                }
            Tier::Enterprise =>
                match self {
                    UsageType::Fast => 10000000,
                    UsageType::HighQuality => 10000000,
                    UsageType::Segment => 10000000,
                    UsageType::Page => 10000000,
                }
            Tier::Starter =>
                match self {
                    UsageType::Fast => 100000,
                    UsageType::HighQuality => 100000,
                    UsageType::Segment => 100000,
                    UsageType::Page => 100000,
                }
            Tier::Dev =>
                match self {
                    UsageType::Fast => 100000,
                    UsageType::HighQuality => 100000,
                    UsageType::Segment => 100000,
                    UsageType::Page => 100000,
                }
            Tier::Team =>
                match self {
                    UsageType::Fast => 25000,
                    UsageType::HighQuality => 25000,
                    UsageType::Segment => 25000,
                    UsageType::Page => 25000,
                }
            Tier::SelfHosted =>
                match self {
                    UsageType::Fast => 10000000,
                    UsageType::HighQuality => 10000000,
                    UsageType::Segment => 10000000,
                    UsageType::Page => 10000000,
                }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema)]
pub struct User {
    pub user_id: String,
    pub customer_id: Option<String>,
    pub email: Option<String>,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub api_keys: Vec<String>,
    pub tier: Tier,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub usage: Vec<UsageLimit>,
    pub task_count: Option<i32>,
    pub last_paid_status: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema, ToSql, FromSql)]
pub struct UsageLimit {
    pub usage_type: UsageType,
    pub usage_limit: i32,
    pub overage_usage: i32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema, ToSql, FromSql)]
pub struct Discount {
    pub usage_type: UsageType,
    pub amount: i32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema, ToSql, FromSql)]
#[postgres(name = "invoice_status")]
pub enum InvoiceStatus {
    Paid,
    Ongoing,
    PastDue,
    Canceled,
    NoInvoice,
    NeedsAction,
    Executed,
}

impl fmt::Display for InvoiceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvoiceStatus::Paid => write!(f, "Paid"),
            InvoiceStatus::Ongoing => write!(f, "ongoing"),
            InvoiceStatus::PastDue => write!(f, "PastDue"),
            InvoiceStatus::Canceled => write!(f, "Canceled"),
            InvoiceStatus::NoInvoice => write!(f, "NoInvoice"),
            InvoiceStatus::NeedsAction => write!(f, "NeedsAction"),
            InvoiceStatus::Executed => write!(f, "Executed"),
        }
    }
}

impl std::str::FromStr for InvoiceStatus {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Paid" => Ok(InvoiceStatus::Paid),
            "ongoing" => Ok(InvoiceStatus::Ongoing),
            "PastDue" => Ok(InvoiceStatus::PastDue),
            "Canceled" => Ok(InvoiceStatus::Canceled),
            "NoInvoice" => Ok(InvoiceStatus::NoInvoice),
            "NeedsAction" => Ok(InvoiceStatus::NeedsAction),
            "Executed" => Ok(InvoiceStatus::Executed),
            _ => Err(()),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema)]
pub struct Usage {
    pub usage: i32,
    pub usage_limit: i32,
    pub usage_type: String,
    pub unit: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
