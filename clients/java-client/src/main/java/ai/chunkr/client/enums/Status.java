package ai.chunkr.client.enums;



/**
 * Enumeration for different task statuses.
 */
public enum Status {
    STARTING("Starting"),
    PROCESSING("Processing"),
    SUCCEEDED("Succeeded"),
    FAILED("Failed"),
    CANCELLED("Cancelled");

    private final String value;

    Status(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    @Override
    public String toString() {
        return value;
    }

    /**
     * Check if this status indicates the task is still in progress.
     * @return true if the task is still processing
     */
    public boolean isProcessing() {
        return this == STARTING || this == PROCESSING;
    }

    /**
     * Check if this status indicates the task has completed (either successfully or with failure).
     * @return true if the task has reached a terminal state
     */
    public boolean isTerminal() {
        return this == SUCCEEDED || this == FAILED || this == CANCELLED;
    }
}