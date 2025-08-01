package ai.chunkr.client.config;

/**
 * Configuration class for the Chunkr client.
 */
public class ClientConfig {
    
    private String apiKey;
    private String baseUrl;

    // Constructors
    public ClientConfig() {
        this.baseUrl = "https://api.chunkr.ai";
    }

    public ClientConfig(String apiKey) {
        this();
        this.apiKey = apiKey;
    }

    public ClientConfig(String apiKey, String baseUrl) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl != null ? baseUrl.replaceAll("/$", "") : "https://api.chunkr.ai";
    }

    // Getters and Setters
    public String getApiKey() {
        return apiKey;
    }

    public void setApiKey(String apiKey) {
        this.apiKey = apiKey;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl != null ? baseUrl.replaceAll("/$", "") : "https://api.chunkr.ai";
    }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String apiKey;
        private String baseUrl;

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        public ClientConfig build() {
            return new ClientConfig(apiKey, baseUrl);
        }
    }

    @Override
    public String toString() {
        return "ClientConfig{" +
                "baseUrl='" + baseUrl + '\'' +
                ", apiKey='" + (apiKey != null ? "***" : "null") + '\'' +
                '}';
    }
}