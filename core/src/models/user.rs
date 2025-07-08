use chrono::{DateTime, Utc};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use std::fmt;
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;
use uuid::Uuid;

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
    ToSchema,
)]
#[postgres(name = "tier")]
pub enum Tier {
    Free,
    PayAsYouGo,
    Enterprise,
    SelfHosted,
    Dev,
    Starter,
    Growth,
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
    FromSql,
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
            Tier::Free => match self {
                UsageType::Fast => 1000,
                UsageType::HighQuality => 500,
                UsageType::Segment => 250,
                UsageType::Page => 1000,
            },
            Tier::PayAsYouGo => match self {
                UsageType::Fast => 1000000,
                UsageType::HighQuality => 1000000,
                UsageType::Segment => 1000000,
                UsageType::Page => 10000000,
            },
            Tier::Enterprise => match self {
                UsageType::Fast => 10000000,
                UsageType::HighQuality => 10000000,
                UsageType::Segment => 10000000,
                UsageType::Page => 10000000,
            },
            Tier::Starter => match self {
                UsageType::Fast => 100000,
                UsageType::HighQuality => 100000,
                UsageType::Segment => 100000,
                UsageType::Page => 100000,
            },
            Tier::Dev => match self {
                UsageType::Fast => 100000,
                UsageType::HighQuality => 100000,
                UsageType::Segment => 100000,
                UsageType::Page => 100000,
            },
            Tier::Growth => match self {
                UsageType::Fast => 25000,
                UsageType::HighQuality => 25000,
                UsageType::Segment => 25000,
                UsageType::Page => 25000,
            },
            Tier::SelfHosted => match self {
                UsageType::Fast => 10000000,
                UsageType::HighQuality => 10000000,
                UsageType::Segment => 10000000,
                UsageType::Page => 10000000,
            },
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema, Default)]
pub enum Status {
    #[default]
    Pending,
    Completed,
}

impl ToSql for Status {
    fn to_sql(
        &self,
        ty: &postgres_types::Type,
        out: &mut postgres_types::private::BytesMut,
    ) -> Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        let s = match self {
            Status::Pending => "Pending",
            Status::Completed => "Completed",
        };
        s.to_sql(ty, out)
    }

    fn accepts(ty: &postgres_types::Type) -> bool {
        <String as ToSql>::accepts(ty)
    }

    postgres_types::to_sql_checked!();
}

impl<'a> FromSql<'a> for Status {
    fn from_sql(
        ty: &postgres_types::Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let s = String::from_sql(ty, raw)?;
        match s.as_str() {
            "Pending" => Ok(Status::Pending),
            "Completed" => Ok(Status::Completed),
            _ => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid Status value: {s}"),
            ))),
        }
    }

    fn accepts(ty: &postgres_types::Type) -> bool {
        <String as FromSql>::accepts(ty)
    }
}

#[derive(
    Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema, Default, ToSql, FromSql,
)]
pub struct Information {
    pub use_case: String,
    pub usage: String,
    pub file_types: String,
    pub referral_source: String,
    pub add_ons: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, ToSchema)]
pub struct OnboardingRecord {
    pub id: String,
    pub status: Status,
    pub information: Information,
}

impl Default for OnboardingRecord {
    fn default() -> Self {
        Self::new()
    }
}

impl OnboardingRecord {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            status: Status::default(),
            information: Information::default(),
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
    pub onboarding_record: Option<OnboardingRecord>,
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
