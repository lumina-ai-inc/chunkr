package ai.chunkr.client.models;



/**
 * Represents a bounding box with coordinates and dimensions.
 */
public class BoundingBox {
    

    private double left;
    

    private double top;
    

    private double width;
    

    private double height;

    // Constructors
    public BoundingBox() {}

    public BoundingBox(double left, double top, double width, double height) {
        this.left = left;
        this.top = top;
        this.width = width;
        this.height = height;
    }

    // Getters and Setters
    public double getLeft() {
        return left;
    }

    public void setLeft(double left) {
        this.left = left;
    }

    public double getTop() {
        return top;
    }

    public void setTop(double top) {
        this.top = top;
    }

    public double getWidth() {
        return width;
    }

    public void setWidth(double width) {
        this.width = width;
    }

    public double getHeight() {
        return height;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    @Override
    public String toString() {
        return "BoundingBox{" +
                "left=" + left +
                ", top=" + top +
                ", width=" + width +
                ", height=" + height +
                '}';
    }
}