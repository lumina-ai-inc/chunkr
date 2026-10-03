package ai.chunkr.client.models;

import ai.chunkr.client.enums.Status;


import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Represents a response from a Chunkr API task operation.
 * Contains methods for polling, updating, and managing task state.
 */
public class TaskResponse {
    
    private String taskId;
    
    private Status status;
    
    private Configuration configuration;
    
    private String createdAt;
    
    private String finishedAt;
    
    private String expiresAt;
    
    private String message;
    
    private Output output;
    
    private String taskUrl;
    
    private String error;

    private Object client; // ChunkrClient - using Object to avoid circular dependency

    // Constructors
    public TaskResponse() {}

    public TaskResponse(Object client) {
        this.client = client;
    }

    // Getters and Setters
    public String getTaskId() {
        return taskId;
    }

    public void setTaskId(String taskId) {
        this.taskId = taskId;
    }

    public Status getStatus() {
        return status;
    }

    public void setStatus(Status status) {
        this.status = status;
    }

    public Configuration getConfiguration() {
        return configuration;
    }

    public void setConfiguration(Configuration configuration) {
        this.configuration = configuration;
    }

    public String getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(String createdAt) {
        this.createdAt = createdAt;
    }

    public String getFinishedAt() {
        return finishedAt;
    }

    public void setFinishedAt(String finishedAt) {
        this.finishedAt = finishedAt;
    }

    public String getExpiresAt() {
        return expiresAt;
    }

    public void setExpiresAt(String expiresAt) {
        this.expiresAt = expiresAt;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Output getOutput() {
        return output;
    }

    public void setOutput(Output output) {
        this.output = output;
    }

    public String getTaskUrl() {
        return taskUrl;
    }

    public void setTaskUrl(String taskUrl) {
        this.taskUrl = taskUrl;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }

    public Object getClient() {
        return client;
    }

    public void setClient(Object client) {
        this.client = client;
    }

    /**
     * Poll the task until it reaches a terminal state (Succeeded, Failed, or Cancelled).
     * @param intervalMs Polling interval in milliseconds (default: 1000)
     * @return The completed task response
     * @throws RuntimeException If the task fails or reaches an unexpected state
     * @throws InterruptedException If the polling is interrupted
     */
    public TaskResponse poll(long intervalMs) throws InterruptedException {
        while (status.isProcessing()) {
            try {
                Thread.sleep(intervalMs);
                // TODO: Implement polling when ChunkrClient is available
                // TaskResponse updated = client.getTask(taskId);
                // Update all fields from the new response
                // this.status = updated.status;
                // this.finishedAt = updated.finishedAt;
                // this.message = updated.message;
                // this.output = updated.output;
                // this.error = updated.error;
                
                // For now, just break to avoid infinite loop  
                System.out.println("Task " + taskId + " status: " + status);
                break;
            } catch (Exception e) {
                System.err.println("Polling error for task " + taskId + ": " + e.getMessage());
                // Continue polling on error
            }
        }

        if (!status.isTerminal()) {
            throw new RuntimeException("Task ended in unexpected state: " + status);
        }

        if (status == Status.FAILED) {
            throw new RuntimeException(error != null ? error : "Task failed without specific error message");
        }

        return this;
    }

    /**
     * Poll the task with default interval of 1000ms.
     * @return The completed task response
     * @throws RuntimeException If the task fails or reaches an unexpected state
     * @throws InterruptedException If the polling is interrupted
     */
    public TaskResponse poll() throws InterruptedException {
        return poll(1000);
    }

    /**
     * Cancel the current task. Only works if the task hasn't started processing.
     * @throws RuntimeException If the task has already started processing
     */
    public void cancel() {
        try {
            // client.cancelTask(taskId); // TODO: Fix when ChunkrClient is available
            poll();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Task cancellation interrupted", e);
        }
    }

    /**
     * Delete the current task.
     * @throws RuntimeException If the task is currently processing
     */
    public void delete() {
        // client.deleteTask(taskId); // TODO: Fix when ChunkrClient is available
    }

    /**
     * Get content from the task's output chunks.
     * @param type The type of content to retrieve ("html", "markdown", or "content")
     * @return The concatenated content of all chunks
     */
    public String getContent(String type) {
        if (output == null || output.getChunks() == null) {
            return "";
        }

        return output.getChunks().stream()
                .flatMap(chunk -> chunk.getSegments().stream())
                .map(segment -> {
                    switch (type.toLowerCase()) {
                        case "html":
                            return segment.getHtml();
                        case "markdown":
                            return segment.getMarkdown();
                        case "content":
                        default:
                            return segment.getContent();
                    }
                })
                .filter(content -> content != null && !content.trim().isEmpty())
                .collect(Collectors.joining("\n"));
    }

    /**
     * Get HTML content from the task's output chunks.
     * @return The concatenated HTML content of all chunks
     */
    public String getHtml() {
        return getContent("html");
    }

    /**
     * Get Markdown content from the task's output chunks.
     * @return The concatenated Markdown content of all chunks
     */
    public String getMarkdown() {
        return getContent("markdown");
    }

    /**
     * Get raw content from the task's output chunks.
     * @return The concatenated raw content of all chunks
     */
    public String getContent() {
        return getContent("content");
    }

    @Override
    public String toString() {
        return "TaskResponse{" +
                "taskId='" + taskId + '\'' +
                ", status=" + status +
                ", createdAt='" + createdAt + '\'' +
                ", finishedAt='" + finishedAt + '\'' +
                ", message='" + message + '\'' +
                ", error='" + error + '\'' +
                '}';
    }
}