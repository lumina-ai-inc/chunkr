package ai.chunkr.client.models;



import java.util.List;

/**
 * Represents the output of a processed document task.
 */
public class Output {
    

    private String fileName;
    

    private String pdfUrl;
    

    private Integer pageCount;
    

    private List<Chunk> chunks;

    // Constructors
    public Output() {}

    public Output(String fileName, String pdfUrl, Integer pageCount, List<Chunk> chunks) {
        this.fileName = fileName;
        this.pdfUrl = pdfUrl;
        this.pageCount = pageCount;
        this.chunks = chunks;
    }

    // Getters and Setters
    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public String getPdfUrl() {
        return pdfUrl;
    }

    public void setPdfUrl(String pdfUrl) {
        this.pdfUrl = pdfUrl;
    }

    public Integer getPageCount() {
        return pageCount;
    }

    public void setPageCount(Integer pageCount) {
        this.pageCount = pageCount;
    }

    public List<Chunk> getChunks() {
        return chunks;
    }

    public void setChunks(List<Chunk> chunks) {
        this.chunks = chunks;
    }

    @Override
    public String toString() {
        return "Output{" +
                "fileName='" + fileName + '\'' +
                ", pdfUrl='" + pdfUrl + '\'' +
                ", pageCount=" + pageCount +
                ", chunks=" + chunks +
                '}';
    }
}