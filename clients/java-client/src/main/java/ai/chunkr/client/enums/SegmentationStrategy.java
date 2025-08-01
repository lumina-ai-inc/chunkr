package ai.chunkr.client.enums;



/**
 * Enumeration for different segmentation strategies used in document processing.
 */
public enum SegmentationStrategy {
    /**
     * Use layout analysis for segmentation
     */
    LAYOUT_ANALYSIS("LayoutAnalysis"),
    
    /**
     * Use page-based segmentation
     */
    PAGE("Page");

    private final String value;

    SegmentationStrategy(String value) {
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