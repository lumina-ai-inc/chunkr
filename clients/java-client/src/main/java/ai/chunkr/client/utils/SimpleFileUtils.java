package ai.chunkr.client.utils;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

/**
 * Simplified utility class for handling file operations without external dependencies.
 * This is a standalone version that can compile without OkHttp dependencies.
 */
public class SimpleFileUtils {
    
    /**
     * Represents a file input that can be uploaded.
     */
    public static class FileInput {
        private final String filename;
        private final byte[] data;
        
        public FileInput(String filename, byte[] data) {
            this.filename = filename;
            this.data = data;
        }
        
        public String getFilename() {
            return filename;
        }
        
        public byte[] getData() {
            return data;
        }
    }

    /**
     * Prepare file input from various sources.
     */
    public static FileInput prepareFile(Object input, String filename) throws IOException {
        if (input instanceof String) {
            String inputStr = (String) input;
            
            // Handle URLs
            if (inputStr.startsWith("http://") || inputStr.startsWith("https://")) {
                return prepareFromUrl(inputStr);
            }
            
            // Handle base64 strings
            if (inputStr.contains(";base64,")) {
                return prepareFromBase64(inputStr);
            }
            
            // Handle file paths
            return prepareFromFilePath(inputStr);
        } else if (input instanceof byte[]) {
            return prepareFromByteArray((byte[]) input, filename);
        } else if (input instanceof InputStream) {
            return prepareFromInputStream((InputStream) input, filename);
        } else if (input instanceof File) {
            return prepareFromFile((File) input);
        } else {
            throw new IllegalArgumentException("Unsupported input type: " + input.getClass().getName());
        }
    }

    private static FileInput prepareFromFilePath(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("File not found: " + filePath);
        }
        
        byte[] data = Files.readAllBytes(path);
        String filename = path.getFileName().toString();
        
        return new FileInput(filename, data);
    }

    private static FileInput prepareFromFile(File file) throws IOException {
        if (!file.exists()) {
            throw new FileNotFoundException("File not found: " + file.getAbsolutePath());
        }
        
        byte[] data = Files.readAllBytes(file.toPath());
        String filename = file.getName();
        
        return new FileInput(filename, data);
    }

    private static FileInput prepareFromUrl(String urlStr) throws IOException {
        URL url = new URL(urlStr);
        
        try (InputStream inputStream = url.openStream();
             ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            
            byte[] data = outputStream.toByteArray();
            String filename = extractFilenameFromUrl(urlStr);
            
            return new FileInput(filename, data);
        }
    }

    private static FileInput prepareFromBase64(String base64String) {
        String[] parts = base64String.split(",", 2);
        if (parts.length != 2) {
            throw new IllegalArgumentException("Invalid base64 format");
        }
        
        String base64Data = parts[1];
        byte[] data = Base64.getDecoder().decode(base64Data);
        
        String filename = "document"; // Default filename for base64 data
        return new FileInput(filename, data);
    }

    private static FileInput prepareFromByteArray(byte[] data, String filename) {
        if (filename == null) {
            filename = "document";
        }
        return new FileInput(filename, data);
    }

    private static FileInput prepareFromInputStream(InputStream inputStream, String filename) throws IOException {
        try (ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            
            byte[] data = outputStream.toByteArray();
            
            if (filename == null) {
                filename = "document";
            }
            
            return new FileInput(filename, data);
        }
    }

    private static String extractFilenameFromUrl(String urlStr) {
        try {
            URL url = new URL(urlStr);
            String path = url.getPath();
            String[] segments = path.split("/");
            String filename = segments[segments.length - 1];
            
            if (filename.isEmpty() || !filename.contains(".")) {
                return "document.pdf";
            }
            
            return filename;
        } catch (Exception e) {
            return "document.pdf";
        }
    }

    public static String getFileExtension(String filename) {
        if (filename == null) return "";
        int lastDotIndex = filename.lastIndexOf('.');
        if (lastDotIndex == -1 || lastDotIndex == filename.length() - 1) {
            return "";
        }
        return filename.substring(lastDotIndex + 1);
    }
}