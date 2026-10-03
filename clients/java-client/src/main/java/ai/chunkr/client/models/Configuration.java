package ai.chunkr.client.models;

import ai.chunkr.client.enums.OcrStrategy;
import ai.chunkr.client.enums.SegmentationStrategy;


/**
 * Main configuration class for Chunkr API requests.
 */

public class Configuration {
    

    private ChunkProcessing chunkProcessing;
    

    private Integer expiresIn;
    

    private Boolean highResolution;
    

    private OcrStrategy ocrStrategy;
    

    private SegmentProcessing segmentProcessing;
    

    private SegmentationStrategy segmentationStrategy;
    

    private String inputFileUrl;

    // Constructors
    public Configuration() {}

    // Getters and Setters
    public ChunkProcessing getChunkProcessing() {
        return chunkProcessing;
    }

    public void setChunkProcessing(ChunkProcessing chunkProcessing) {
        this.chunkProcessing = chunkProcessing;
    }

    public Integer getExpiresIn() {
        return expiresIn;
    }

    public void setExpiresIn(Integer expiresIn) {
        this.expiresIn = expiresIn;
    }

    public Boolean getHighResolution() {
        return highResolution;
    }

    public void setHighResolution(Boolean highResolution) {
        this.highResolution = highResolution;
    }

    public OcrStrategy getOcrStrategy() {
        return ocrStrategy;
    }

    public void setOcrStrategy(OcrStrategy ocrStrategy) {
        this.ocrStrategy = ocrStrategy;
    }

    public SegmentProcessing getSegmentProcessing() {
        return segmentProcessing;
    }

    public void setSegmentProcessing(SegmentProcessing segmentProcessing) {
        this.segmentProcessing = segmentProcessing;
    }

    public SegmentationStrategy getSegmentationStrategy() {
        return segmentationStrategy;
    }

    public void setSegmentationStrategy(SegmentationStrategy segmentationStrategy) {
        this.segmentationStrategy = segmentationStrategy;
    }

    public String getInputFileUrl() {
        return inputFileUrl;
    }

    public void setInputFileUrl(String inputFileUrl) {
        this.inputFileUrl = inputFileUrl;
    }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private ChunkProcessing chunkProcessing;
        private Integer expiresIn;
        private Boolean highResolution;
        private OcrStrategy ocrStrategy;
        private SegmentProcessing segmentProcessing;
        private SegmentationStrategy segmentationStrategy;
        private String inputFileUrl;

        public Builder chunkProcessing(ChunkProcessing chunkProcessing) {
            this.chunkProcessing = chunkProcessing;
            return this;
        }

        public Builder expiresIn(Integer expiresIn) {
            this.expiresIn = expiresIn;
            return this;
        }

        public Builder highResolution(Boolean highResolution) {
            this.highResolution = highResolution;
            return this;
        }

        public Builder ocrStrategy(OcrStrategy ocrStrategy) {
            this.ocrStrategy = ocrStrategy;
            return this;
        }

        public Builder segmentProcessing(SegmentProcessing segmentProcessing) {
            this.segmentProcessing = segmentProcessing;
            return this;
        }

        public Builder segmentationStrategy(SegmentationStrategy segmentationStrategy) {
            this.segmentationStrategy = segmentationStrategy;
            return this;
        }

        public Builder inputFileUrl(String inputFileUrl) {
            this.inputFileUrl = inputFileUrl;
            return this;
        }

        public Configuration build() {
            Configuration config = new Configuration();
            config.chunkProcessing = chunkProcessing;
            config.expiresIn = expiresIn;
            config.highResolution = highResolution;
            config.ocrStrategy = ocrStrategy;
            config.segmentProcessing = segmentProcessing;
            config.segmentationStrategy = segmentationStrategy;
            config.inputFileUrl = inputFileUrl;
            return config;
        }
    }
}