package ai.chunkr.client.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;

/**
 * Utility class for JSON operations and creating form data.
 */
public class JsonUtils {
    
    private static final ObjectMapper objectMapper;
    
    static {
        objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        objectMapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
    }

    /**
     * Get the configured ObjectMapper instance.
     * @return ObjectMapper configured for the Chunkr API
     */
    public static ObjectMapper getObjectMapper() {
        return objectMapper;
    }

    /**
     * Convert an object to JSON string.
     * @param object The object to serialize
     * @return JSON string representation
     * @throws RuntimeException If serialization fails
     */
    public static String toJson(Object object) {
        try {
            return objectMapper.writeValueAsString(object);
        } catch (Exception e) {
            throw new RuntimeException("Failed to serialize object to JSON", e);
        }
    }

    /**
     * Parse JSON string to object.
     * @param json The JSON string
     * @param clazz The target class
     * @param <T> The type of the target class
     * @return Deserialized object
     * @throws RuntimeException If deserialization fails
     */
    public static <T> T fromJson(String json, Class<T> clazz) {
        try {
            return objectMapper.readValue(json, clazz);
        } catch (Exception e) {
            throw new RuntimeException("Failed to deserialize JSON to object", e);
        }
    }

    /**
     * Create a multipart body part for JSON data.
     * @param key The form field name
     * @param object The object to serialize as JSON
     * @return MultipartBody.Part containing the JSON data
     */
    public static MultipartBody.Part createJsonPart(String key, Object object) {
        String json = toJson(object);
        RequestBody requestBody = RequestBody.create(json, MediaType.parse("application/json"));
        return MultipartBody.Part.createFormData(key, json);
    }

    /**
     * Create a simple text form data part.
     * @param key The form field name
     * @param value The text value
     * @return MultipartBody.Part containing the text data
     */
    public static MultipartBody.Part createTextPart(String key, String value) {
        RequestBody requestBody = RequestBody.create(value, MediaType.parse("text/plain"));
        return MultipartBody.Part.createFormData(key, value);
    }
}