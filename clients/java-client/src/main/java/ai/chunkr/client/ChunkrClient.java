package ai.chunkr.client;

import ai.chunkr.client.config.ClientConfig;
import ai.chunkr.client.models.Configuration;
import ai.chunkr.client.models.TaskResponse;
import ai.chunkr.client.utils.FileUtils;
import ai.chunkr.client.utils.JsonUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Main client class for interacting with the Chunkr API.
 * Provides methods for uploading documents, managing tasks, and retrieving results.
 * 
 * Example usage:
 * <pre>
 * {@code
 * ChunkrClient client = new ChunkrClient("your-api-key");
 * 
 * // Upload and process a document
 * TaskResponse task = client.upload("document.pdf");
 * 
 * // Get the results
 * String html = task.getHtml();
 * String markdown = task.getMarkdown();
 * 
 * // Clean up
 * client.close();
 * }
 * </pre>
 */
public class ChunkrClient {
    
    private static final Logger logger = LoggerFactory.getLogger(ChunkrClient.class);
    
    private final ClientConfig config;
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;

    /**
     * Create a new Chunkr client using environment variables.
     * Looks for CHUNKR_API_KEY and optionally CHUNKR_URL environment variables.
     * @throws IllegalArgumentException If API key is not found in environment
     */
    public ChunkrClient() {
        this(createConfigFromEnvironment());
    }

    /**
     * Create a new Chunkr client with the specified API key.
     * @param apiKey The API key for authentication
     */
    public ChunkrClient(String apiKey) {
        this(new ClientConfig(apiKey));
    }

    /**
     * Create a new Chunkr client with the specified API key and base URL.
     * @param apiKey The API key for authentication
     * @param baseUrl The base URL for the API
     */
    public ChunkrClient(String apiKey, String baseUrl) {
        this(new ClientConfig(apiKey, baseUrl));
    }

    /**
     * Create a new Chunkr client with the specified configuration.
     * @param config The client configuration
     */
    public ChunkrClient(ClientConfig config) {
        if (config.getApiKey() == null || config.getApiKey().trim().isEmpty()) {
            throw new IllegalArgumentException(
                "API key must be provided either directly, in .env file, or as CHUNKR_API_KEY environment variable. " +
                "You can get an API key at: https://www.chunkr.ai"
            );
        }
        
        this.config = config;
        this.objectMapper = JsonUtils.getObjectMapper();
        this.httpClient = createHttpClient();
        
        logger.info("Initialized Chunkr client with base URL: {}", config.getBaseUrl());
    }

    /**
     * Create configuration from environment variables.
     */
    private static ClientConfig createConfigFromEnvironment() {
        String apiKey = System.getenv("CHUNKR_API_KEY");
        String baseUrl = System.getenv("CHUNKR_URL");
        
        return new ClientConfig(apiKey, baseUrl);
    }

    /**
     * Create and configure the HTTP client.
     */
    private OkHttpClient createHttpClient() {
        return new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(120, TimeUnit.SECONDS)
                .addInterceptor(new AuthenticationInterceptor(config.getApiKey()))
                .addInterceptor(new LoggingInterceptor())
                .build();
    }

    /**
     * Upload a file and wait for processing to complete.
     * @param file The file to upload (can be a file path, URL, byte array, etc.)
     * @return The completed task response
     * @throws RuntimeException If upload or processing fails
     */
    public TaskResponse upload(Object file) {
        return upload(file, null);
    }

    /**
     * Upload a file with configuration and wait for processing to complete.
     * @param file The file to upload (can be a file path, URL, byte array, etc.)
     * @param config Optional configuration for processing
     * @return The completed task response
     * @throws RuntimeException If upload or processing fails
     */
    public TaskResponse upload(Object file, Configuration config) {
        try {
            TaskResponse task = createTask(file, config);
            return task.poll();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Upload interrupted", e);
        }
    }

    /**
     * Create a new task without waiting for completion.
     * @param file The file to upload
     * @return The initial task response
     * @throws RuntimeException If task creation fails
     */
    public TaskResponse createTask(Object file) {
        return createTask(file, null);
    }

    /**
     * Create a new task with configuration without waiting for completion.
     * @param file The file to upload
     * @param config Optional configuration for processing
     * @return The initial task response
     * @throws RuntimeException If task creation fails
     */
    public TaskResponse createTask(Object file, Configuration config) {
        try {
            FileUtils.FileInput fileInput = FileUtils.prepareFile(file, null);
            
            MultipartBody.Builder builder = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addPart(FileUtils.createFilePart(fileInput));
            
            if (config != null) {
                builder.addPart(JsonUtils.createJsonPart("configuration", config));
            }
            
            RequestBody requestBody = builder.build();
            
            Request request = new Request.Builder()
                    .url(config.getBaseUrl() + "/api/v1/task")
                    .post(requestBody)
                    .build();
            
            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Task creation failed: " + response.code() + " " + response.message());
                }
                
                String responseBody = response.body().string();
                TaskResponse taskResponse = objectMapper.readValue(responseBody, TaskResponse.class);
                taskResponse.setClient(this);
                
                return taskResponse;
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create task", e);
        }
    }

    /**
     * Get a task by its ID.
     * @param taskId The ID of the task to retrieve
     * @return The task response
     * @throws RuntimeException If retrieval fails
     */
    public TaskResponse getTask(String taskId) {
        try {
            Request request = new Request.Builder()
                    .url(config.getBaseUrl() + "/api/v1/task/" + taskId)
                    .get()
                    .build();
            
            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Failed to get task: " + response.code() + " " + response.message());
                }
                
                String responseBody = response.body().string();
                TaskResponse taskResponse = objectMapper.readValue(responseBody, TaskResponse.class);
                taskResponse.setClient(this);
                
                return taskResponse;
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to get task", e);
        }
    }

    /**
     * Update a task with new configuration.
     * @param taskId The ID of the task to update
     * @param config The new configuration
     * @return The updated task response
     * @throws RuntimeException If update fails
     */
    public TaskResponse updateTask(String taskId, Configuration config) {
        try {
            MultipartBody.Builder builder = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM);
            
            if (config != null) {
                builder.addPart(JsonUtils.createJsonPart("configuration", config));
            }
            
            RequestBody requestBody = builder.build();
            
            Request request = new Request.Builder()
                    .url(this.config.getBaseUrl() + "/api/v1/task/" + taskId)
                    .patch(requestBody)
                    .build();
            
            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Task update failed: " + response.code() + " " + response.message());
                }
                
                String responseBody = response.body().string();
                TaskResponse taskResponse = objectMapper.readValue(responseBody, TaskResponse.class);
                taskResponse.setClient(this);
                
                return taskResponse;
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to update task", e);
        }
    }

    /**
     * Delete a task by its ID.
     * @param taskId The ID of the task to delete
     * @throws RuntimeException If deletion fails
     */
    public void deleteTask(String taskId) {
        try {
            Request request = new Request.Builder()
                    .url(config.getBaseUrl() + "/api/v1/task/" + taskId)
                    .delete()
                    .build();
            
            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Failed to delete task: " + response.code() + " " + response.message());
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to delete task", e);
        }
    }

    /**
     * Cancel a task by its ID.
     * @param taskId The ID of the task to cancel
     * @throws RuntimeException If cancellation fails
     */
    public void cancelTask(String taskId) {
        try {
            Request request = new Request.Builder()
                    .url(config.getBaseUrl() + "/api/v1/task/" + taskId + "/cancel")
                    .get()
                    .build();
            
            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Failed to cancel task: " + response.code() + " " + response.message());
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to cancel task", e);
        }
    }

    /**
     * Close the client and release resources.
     */
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
        
        if (httpClient.cache() != null) {
            try {
                httpClient.cache().close();
            } catch (IOException e) {
                logger.warn("Failed to close HTTP cache", e);
            }
        }
    }

    /**
     * Authentication interceptor for adding API key to requests.
     */
    private static class AuthenticationInterceptor implements Interceptor {
        private final String apiKey;

        public AuthenticationInterceptor(String apiKey) {
            this.apiKey = apiKey;
        }

        @Override
        public Response intercept(Chain chain) throws IOException {
            Request original = chain.request();
            Request.Builder requestBuilder = original.newBuilder()
                    .header("Authorization", apiKey);

            Request request = requestBuilder.build();
            return chain.proceed(request);
        }
    }

    /**
     * Logging interceptor for debugging HTTP requests.
     */
    private static class LoggingInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            logger.debug("Sending request: {} {}", request.method(), request.url());
            
            Response response = chain.proceed(request);
            logger.debug("Received response: {} {}", response.code(), response.message());
            
            return response;
        }
    }
}