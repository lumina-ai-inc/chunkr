package ai.chunkr.client.enums;



/**
 * Enumeration for different generation strategies used in document processing.
 */
public enum GenerationStrategy {
    /**
     * Use Language Learning Model for generation
     */
    LLM("LLM"),
    
    /**
     * Use automatic generation strategy
     */
    AUTO("Auto");

    private final String value;

    GenerationStrategy(String value) {
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