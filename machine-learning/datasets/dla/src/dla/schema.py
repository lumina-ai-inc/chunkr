from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional 

class DocumentDomain(str, Enum):
    FINANCIAL = "financial" # e.g., Annual reports, SEC filings, bank statements, loan applications, investment prospectuses.
    BILLING = "billing" # e.g., Invoices, purchase orders, receipts, utility bills, account statements.
    TAX = "tax" # e.g., Tax forms (W2, 1040, etc.), tax returns, official tax documents.
    SUPPLY_CHAIN = "supply_chain" # e.g., Bills of lading, packing slips, shipping manifests, inventory reports, logistics documents, POs, PODs etc.
    TECHNICAL = "technical" # e.g., Manuals, specifications, datasheets, engineering drawings, code documentation.
    RESEARCH = "research" # e.g., Academic papers, scientific articles, conference proceedings, theses, study reports (excluding patents).
    LEGAL = "legal" # e.g., Contracts, court filings, legal briefs, NDAs, terms of service, deeds, wills.
    GOVERNMENT = "government" # e.g., Official forms, regulations, public notices, legislative documents, census forms, government reports.
    PROCUREMENT = "procurement" # e.g., Requests for Proposal (RFP), Requests for Information (RFI), Requests for Quotation (RFQ), Bids, Proposals, Grant Solicitations, Statements of Work (SOW.
    CONSULTING = "consulting" # e.g., Presentations (slides), proposals, reports, case studies prepared by consulting firms.
    MAGAZINE = "magazine" # e.g., Articles, layouts typical of popular or trade magazines (often multi-column, image-heavy).
    NEWSPAPER = "newspaper" # e.g., News articles, editorials, classifieds, typical newspaper layouts.
    TEXTBOOK = "textbook" # e.g., Chapters from educational books, often with diagrams, exercises, specific formatting.
    HISTORICAL = "historical" # e.g., Archived documents, letters, manuscripts, old records, documents clearly pre-modern era.
    PATENT = "patent" # e.g., Official patent filings/grants with specific structure (abstract, claims, drawings).
    EDUCATION = "education" # e.g., Homework assignments, exam papers, syllabi, lecture notes, educational worksheets (distinct from textbooks).
    MEDICAL = "medical" # e.g., Patient records, medical charts, prescriptions, hospital forms, medical billing (use 'billing' if primarily an invoice), EOBs (Explanation of Benefits). Excludes general research papers (use 'research').
    REAL_ESTATE = "real_estate" # e.g., Property listings, deeds (use 'legal' if primarily a legal contract), appraisal reports, lease agreements, MLS sheets.
    CONSTRUCTION = "construction" # e.g., Blueprints, architectural drawings, construction plans, permits, inspection reports, change orders, material lists, safety manuals specific to construction sites.
    MISCELLANEOUS = "miscellaneous" # e.g., Identification cards, resumes, certificates, flyers, brochures, menus, general correspondence not fitting other categories.
    UNKNOWN = "unknown" # Fallback if no other category strongly fits.
    
class DocumentSource(str, Enum):
    """Describes the origin or method of creation/capture of the document page."""
    NATIVE_DIGITAL = "native_digital" # Born-digital (e.g., exported PDF, Word doc)
    SCANNED_CLEAN = "scanned_clean"   # Scanned from high-quality print
    SCANNED_DEGRADED = "scanned_degraded" # Scanned from fax, old paper, noisy source
    PHOTO = "photo"                 # Captured via camera (phone, etc.)
    SCREENSHOT = "screenshot"         # Captured from a screen display
    UNKNOWN = "unknown"             # Source cannot be determined


class Orientation(str, Enum):
    """The intended orientation of the page content."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    MIXED = "mixed" # Contains elements in both orientations significantly
    UNKNOWN = "unknown"


class ColorMode(str, Enum):
    """The color representation of the document page image."""
    COLOR = "color"
    GRAYSCALE = "grayscale"
    BLACK_WHITE = "black_and_white" # Strictly two-tone
    UNKNOWN = "unknown"

    
class Complexity(str, Enum):
    SIMPLE = "Simple" # Minimal variation in layout, few elements, straightforward text flow.
    MEDIUM = "Medium" # Moderate number of elements, standard layouts (e.g., single/double column with images), clear structure.
    COMPLEX = "Complex" # Multiple columns, complex tables, nested elements, varied formatting, significant non-text content.
    VERY_COMPLEX = "Very Complex" # Highly irregular layout, dense information, overlapping elements, unusual structures (e.g., complex forms, dense technical diagrams).

class SegmentType(str, Enum):
    TITLE = "Title" # Main title of the document or a major section.
    HEADING = "Heading" # Section or subsection heading, larger than body text.
    SUBHEADING = "Subheading" # Lower-level heading, smaller than Heading but distinct from body text.
    CODE_BLOCK = "Code Block" # Formatted block of source code.
    LIST_ITEM = "List Item (Bulleted/Numbered)" # Individual item within a bulleted or numbered list.
    BLOCK_QUOTE = "Block Quote/Pull Quote" # Indented or specially formatted block of quoted text.
    CAPTION = "Caption" # Text describing a figure, table, or image.
    LEGEND = "Legend" # Key explaining symbols, colors, or patterns used in a visual element (e.g., chart, map, diagram). Distinct from a general caption.
    FORMULA = "Formula" # Mathematical or chemical formula, often specially formatted or typeset.
    HEADER = "Header" # Repeating content at the top of the page (e.g., document title, chapter name).
    FOOTER = "Footer" # Repeating content at the bottom of the page (e.g., confidentiality notice, document version).
    PAGE_NUMBER = "Page Number" # Number indicating the page sequence.
    FOOTNOTE = "Footnote/Endnote" # Ancillary information placed at the bottom of the page or end of a section/document.
    PICTURE = "Picture/Figure/Image/Chart" # Visual elements like photographs, diagrams, charts, graphs.
    TABLE = "Table" # Data organized in rows and columns.
    TEXT_BLOCK = "Text Block/Paragraph" # Standard block of running text, the main content carrier.
    FORM_REGION = "Form Region/Key-Value Pair" # Areas designed for data entry, often with labels and fields (e.g., key-value pairs).
    SIGNATURE = "Signature/Signature Block" # Handwritten signature or designated area for one.
    HANDWRITING = "Handwriting" # Significant portions of handwritten text (not just signatures).
    GRAPHICAL_ITEM = "Graphical Item (Logo, QR, Barcode, Stamp)" # Small graphical elements like logos, barcodes, QR codes, official stamps.
    UNKNOWN = "Unknown/Other" # Segment type that doesn't fit into the above categories.


class LayoutStyle(str, Enum):
    SINGLE_COLUMN = "Single Column" # Text flows in one main column.
    DOUBLE_COLUMN = "Double Column" # Text flows in two distinct columns (common in papers, magazines).
    MULTI_COLUMN = "Multiple Columns" # Text flows in three or more distinct columns (common in newspapers, some brochures).
    COMPLEX = "Complex/Mixed" # Combination of different column layouts, non-standard flow, or highly irregular structure.
    IMAGE_HEAVY = "Image Heavy" # Dominated by images/figures with relatively less text.
    FORM_LIKE = "Form-like" # Primarily structured as a form with fields and labels.
    UNKNOWN = "Unknown" # Layout style cannot be determined or doesn't fit categories.


class LanguageCode(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    DUTCH = "nl"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    POLISH = "pl"
    SWEDISH = "sv"
    TURKISH = "tr"
    GREEK = "el"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HEBREW = "he"
    HINDI = "hi"
    BENGALI = "bn"
    URDU = "ur"
    TAMIL = "ta"
    PERSIAN = "fa"
    VIETNAMESE = "vi"
    THAI = "th"
    INDONESIAN = "id"
    MALAY = "ms"
    TAGALOG = "tl"
    MIXED = "mixed"
    OTHER = "other"
    UNKNOWN = "unknown"


class DefectType(str, Enum):
    SKEW = "skew" # Page is rotated or slanted.
    BLUR = "blur" # Text or image elements are out of focus or low resolution.
    WATERMARK = "watermark" # Overlay (text or image) partially obscuring content.
    STAIN = "stain" # Discoloration from liquids, dirt, etc.
    CREASE = "crease" # Visible fold lines on the page.
    HOLE_PUNCH = "hole_punch" # Holes from binder punching.
    LOW_CONTRAST = "low_contrast" # Text is faint or background interferes with readability.
    CUT_OFF = "cut_off" # Edges of the document content are missing.
    SCANNED = "scanned" # Artifacts typical of scanning (e.g., scanner bed edges, slight noise, minor distortions not covered by skew/blur).


class PageProfile(BaseModel):
    domain: DocumentDomain = Field()
    source: DocumentSource = Field()
    orientation: Orientation = Field()
    color_mode: ColorMode = Field()
    complexity: Complexity = Field()
    layout_style: LayoutStyle = Field()
    primary_language: LanguageCode = Field(description="The primary language used on the page. Use 'mixed' if multiple significant languages are present, 'other' for less common languages, 'unknown' if indeterminable.")
    segments: List[SegmentType] = Field()
    defects: List[DefectType] = Field()
    is_sparse_or_empty: bool = Field(description="True if the page has very little meaningful content (e.g., blank page, only header/footer/page number, separator page).")
    page_identifier: Optional[str] = Field(description="Unique identifier (dedup_id) added during processing.")

    model_config = {
        "json_schema_extra": {
            "additionalProperties": False,
        }
    }
    
    