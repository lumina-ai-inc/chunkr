package ai.chunkr.client.models;

import ai.chunkr.client.enums.SegmentType;


import java.util.List;

/**
 * Represents a document segment with its properties and content.
 */
public class Segment {
    

    private String segmentId;
    

    private BoundingBox bbox;
    

    private int pageNumber;
    

    private double pageWidth;
    

    private double pageHeight;
    

    private String content;
    

    private SegmentType segmentType;
    

    private List<OCRResult> ocr;
    

    private String image;
    

    private String html;
    

    private String markdown;

    // Constructors
    public Segment() {}

    // Getters and Setters
    public String getSegmentId() {
        return segmentId;
    }

    public void setSegmentId(String segmentId) {
        this.segmentId = segmentId;
    }

    public BoundingBox getBbox() {
        return bbox;
    }

    public void setBbox(BoundingBox bbox) {
        this.bbox = bbox;
    }

    public int getPageNumber() {
        return pageNumber;
    }

    public void setPageNumber(int pageNumber) {
        this.pageNumber = pageNumber;
    }

    public double getPageWidth() {
        return pageWidth;
    }

    public void setPageWidth(double pageWidth) {
        this.pageWidth = pageWidth;
    }

    public double getPageHeight() {
        return pageHeight;
    }

    public void setPageHeight(double pageHeight) {
        this.pageHeight = pageHeight;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public SegmentType getSegmentType() {
        return segmentType;
    }

    public void setSegmentType(SegmentType segmentType) {
        this.segmentType = segmentType;
    }

    public List<OCRResult> getOcr() {
        return ocr;
    }

    public void setOcr(List<OCRResult> ocr) {
        this.ocr = ocr;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }

    public String getHtml() {
        return html;
    }

    public void setHtml(String html) {
        this.html = html;
    }

    public String getMarkdown() {
        return markdown;
    }

    public void setMarkdown(String markdown) {
        this.markdown = markdown;
    }

    @Override
    public String toString() {
        return "Segment{" +
                "segmentId='" + segmentId + '\'' +
                ", bbox=" + bbox +
                ", pageNumber=" + pageNumber +
                ", pageWidth=" + pageWidth +
                ", pageHeight=" + pageHeight +
                ", content='" + content + '\'' +
                ", segmentType=" + segmentType +
                ", ocr=" + ocr +
                ", image='" + image + '\'' +
                ", html='" + html + '\'' +
                ", markdown='" + markdown + '\'' +
                '}';
    }
}