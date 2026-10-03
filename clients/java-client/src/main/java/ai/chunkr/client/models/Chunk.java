package ai.chunkr.client.models;



import java.util.List;

/**
 * Represents a chunk containing multiple document segments.
 */
public class Chunk {
    

    private String chunkId;
    

    private int chunkLength;
    

    private List<Segment> segments;

    // Constructors
    public Chunk() {}

    public Chunk(String chunkId, int chunkLength, List<Segment> segments) {
        this.chunkId = chunkId;
        this.chunkLength = chunkLength;
        this.segments = segments;
    }

    // Getters and Setters
    public String getChunkId() {
        return chunkId;
    }

    public void setChunkId(String chunkId) {
        this.chunkId = chunkId;
    }

    public int getChunkLength() {
        return chunkLength;
    }

    public void setChunkLength(int chunkLength) {
        this.chunkLength = chunkLength;
    }

    public List<Segment> getSegments() {
        return segments;
    }

    public void setSegments(List<Segment> segments) {
        this.segments = segments;
    }

    @Override
    public String toString() {
        return "Chunk{" +
                "chunkId='" + chunkId + '\'' +
                ", chunkLength=" + chunkLength +
                ", segments=" + segments +
                '}';
    }
}