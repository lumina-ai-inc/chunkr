package ai.chunkr.client.models;



/**
 * Configuration for chunk processing settings.
 */

public class ChunkProcessing {
    

    private Integer targetLength;

    // Constructors
    public ChunkProcessing() {}

    public ChunkProcessing(Integer targetLength) {
        this.targetLength = targetLength;
    }

    // Getters and Setters
    public Integer getTargetLength() {
        return targetLength;
    }

    public void setTargetLength(Integer targetLength) {
        this.targetLength = targetLength;
    }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Integer targetLength;

        public Builder targetLength(Integer targetLength) {
            this.targetLength = targetLength;
            return this;
        }

        public ChunkProcessing build() {
            return new ChunkProcessing(targetLength);
        }
    }
}