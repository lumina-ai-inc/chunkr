package ai.chunkr.client.utils;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

/**
 * Utility class for handling file operations and preparing file data for upload.
 */
public class FileUtils {
    
    /**
     * Represents a file input that can be uploaded.
     */
    public static class FileInput {
        private final String filename;
        private final byte[] data;
        private final MediaType mediaType;
        
        public FileInput(String filename, byte[] data, MediaType mediaType) {
            this.filename = filename;
            this.data = data;
            this.mediaType = mediaType;
        }
        
        public String getFilename() {
            return filename;
        }
        
        public byte[] getData() {
            return data;
        }
        
        public MediaType getMediaType() {
            return mediaType;
        }
    }

    /**
     * Prepare file input from various sources (file path, URL, byte array, InputStream).
     * @param input The input source (String path/URL, byte[], or InputStream)
     * @param filename Optional filename (used for byte[] and InputStream inputs)
     * @return FileInput object ready for upload
     * @throws IOException If file processing fails
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

    /**
     * Prepare file input from a file path.
     */
    private static FileInput prepareFromFilePath(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("File not found: " + filePath);
        }
        
        byte[] data = Files.readAllBytes(path);
        String filename = path.getFileName().toString();
        MediaType mediaType = determineMediaType(filename);
        
        return new FileInput(filename, data, mediaType);
    }

    /**
     * Prepare file input from a File object.
     */
    private static FileInput prepareFromFile(File file) throws IOException {
        if (!file.exists()) {
            throw new FileNotFoundException("File not found: " + file.getAbsolutePath());
        }
        
        byte[] data = Files.readAllBytes(file.toPath());
        String filename = file.getName();
        MediaType mediaType = determineMediaType(filename);
        
        return new FileInput(filename, data, mediaType);
    }

    /**
     * Prepare file input from a URL.
     */
    private static FileInput prepareFromUrl(String urlStr) throws IOException {
        URL url = new URL(urlStr);
        
        // Download the file
        try (InputStream inputStream = url.openStream();
             ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            
            byte[] data = outputStream.toByteArray();
            
            // Extract filename from URL
            String filename = extractFilenameFromUrl(urlStr);
            MediaType mediaType = determineMediaType(filename);
            
            return new FileInput(filename, data, mediaType);
        }
    }

    /**
     * Prepare file input from a base64 string.
     */
    private static FileInput prepareFromBase64(String base64String) {
        String[] parts = base64String.split(",", 2);
        if (parts.length != 2) {
            throw new IllegalArgumentException("Invalid base64 format");
        }
        
        String header = parts[0];
        String base64Data = parts[1];
        
        byte[] data = Base64.getDecoder().decode(base64Data);
        
        // Extract MIME type and map to extension
        String mimeType = extractMimeType(header);
        String extension = mapMimeTypeToExtension(mimeType);
        String filename = "document." + extension;
        MediaType mediaType = MediaType.parse(mimeType);
        
        return new FileInput(filename, data, mediaType);
    }

    /**
     * Prepare file input from a byte array.
     */
    private static FileInput prepareFromByteArray(byte[] data, String filename) {
        if (filename == null) {
            filename = "document";
        }
        
        MediaType mediaType = determineMediaType(filename);
        return new FileInput(filename, data, mediaType);
    }

    /**
     * Prepare file input from an InputStream.
     */
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
            
            MediaType mediaType = determineMediaType(filename);
            return new FileInput(filename, data, mediaType);
        }
    }

    /**
     * Extract filename from URL.
     */
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

    /**
     * Extract MIME type from base64 header.
     */
    private static String extractMimeType(String header) {
        // Expected format: "data:application/pdf;base64"
        String[] parts = header.split(":");
        if (parts.length >= 2) {
            String mimeTypePart = parts[1].split(";")[0];
            return mimeTypePart.toLowerCase();
        }
        return "application/octet-stream";
    }

    /**
     * Map MIME type to file extension.
     */
    private static String mapMimeTypeToExtension(String mimeType) {
        switch (mimeType) {
            case "application/pdf":
                return "pdf";
            case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            case "application/msword":
                return "docx";
            case "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            case "application/vnd.ms-powerpoint":
                return "pptx";
            case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            case "application/vnd.ms-excel":
                return "xlsx";
            case "image/jpeg":
            case "image/jpg":
                return "jpg";
            case "image/png":
                return "png";
            default:
                return "bin";
        }
    }

    /**
     * Determine MediaType from filename.
     */
    private static MediaType determineMediaType(String filename) {
        if (filename == null) {
            return MediaType.parse("application/octet-stream");
        }
        
        String extension = getFileExtension(filename).toLowerCase();
        
        switch (extension) {
            case "pdf":
                return MediaType.parse("application/pdf");
            case "docx":
                return MediaType.parse("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
            case "doc":
                return MediaType.parse("application/msword");
            case "pptx":
                return MediaType.parse("application/vnd.openxmlformats-officedocument.presentationml.presentation");
            case "ppt":
                return MediaType.parse("application/vnd.ms-powerpoint");
            case "xlsx":
                return MediaType.parse("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
            case "xls":
                return MediaType.parse("application/vnd.ms-excel");
            case "jpg":
            case "jpeg":
                return MediaType.parse("image/jpeg");
            case "png":
                return MediaType.parse("image/png");
            default:
                return MediaType.parse("application/octet-stream");
        }
    }

    /**
     * Get file extension from filename.
     */
    private static String getFileExtension(String filename) {
        int lastDotIndex = filename.lastIndexOf('.');
        if (lastDotIndex == -1 || lastDotIndex == filename.length() - 1) {
            return "";
        }
        return filename.substring(lastDotIndex + 1);
    }

    /**
     * Create a multipart body part for file upload.
     * @param fileInput The file input to create the part for
     * @return MultipartBody.Part ready for upload
     */
    public static MultipartBody.Part createFilePart(FileInput fileInput) {
        RequestBody fileBody = RequestBody.create(fileInput.getData(), fileInput.getMediaType());
        return MultipartBody.Part.createFormData("file", fileInput.getFilename(), fileBody);
    }
}