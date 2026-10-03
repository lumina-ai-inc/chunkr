package ai.chunkr.client.enums;



/**
 * Enumeration for different types of document segments.
 */
public enum SegmentType {
    CAPTION("Caption"),
    FOOTNOTE("Footnote"),
    FORMULA("Formula"),
    LIST_ITEM("ListItem"),
    PAGE("Page"),
    PAGE_FOOTER("PageFooter"),
    PAGE_HEADER("PageHeader"),
    PICTURE("Picture"),
    SECTION_HEADER("SectionHeader"),
    TABLE("Table"),
    TEXT("Text"),
    TITLE("Title");

    private final String value;

    SegmentType(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    @Override
    public String toString() {
        return value;
    }
}