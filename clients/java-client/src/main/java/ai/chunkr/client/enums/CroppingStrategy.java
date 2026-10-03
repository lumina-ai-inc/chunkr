package ai.chunkr.client.enums;



/**
 * Enumeration for different cropping strategies used in document processing.
 */
public enum CroppingStrategy {
    /**
     * Crop all elements
     */
    ALL("All"),
    
    /**
     * Use automatic cropping strategy
     */
    AUTO("Auto");

    private final String value;

    CroppingStrategy(String value) {
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