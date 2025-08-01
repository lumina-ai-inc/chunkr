package ai.chunkr.client.enums;



/**
 * Enumeration for different OCR strategies used in document processing.
 */
public enum OcrStrategy {
    /**
     * Apply OCR to all elements
     */
    ALL("All"),
    
    /**
     * Use automatic OCR strategy
     */
    AUTO("Auto");

    private final String value;

    OcrStrategy(String value) {
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