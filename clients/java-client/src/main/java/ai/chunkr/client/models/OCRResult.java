package ai.chunkr.client.models;



/**
 * Represents the result of OCR processing on a document segment.
 */
public class OCRResult {
    

    private BoundingBox bbox;
    

    private String text;
    

    private Double confidence;

    // Constructors
    public OCRResult() {}

    public OCRResult(BoundingBox bbox, String text, Double confidence) {
        this.bbox = bbox;
        this.text = text;
        this.confidence = confidence;
    }

    // Getters and Setters
    public BoundingBox getBbox() {
        return bbox;
    }

    public void setBbox(BoundingBox bbox) {
        this.bbox = bbox;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    @Override
    public String toString() {
        return "OCRResult{" +
                "bbox=" + bbox +
                ", text='" + text + '\'' +
                ", confidence=" + confidence +
                '}';
    }
}