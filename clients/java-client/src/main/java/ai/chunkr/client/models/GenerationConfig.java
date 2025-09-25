package ai.chunkr.client.models;

import ai.chunkr.client.enums.CroppingStrategy;
import ai.chunkr.client.enums.GenerationStrategy;


/**
 * Configuration for generation settings.
 */

public class GenerationConfig {
    

    private GenerationStrategy html;
    

    private String llm;
    

    private GenerationStrategy markdown;
    

    private CroppingStrategy cropImage;

    // Constructors
    public GenerationConfig() {}

    public GenerationConfig(GenerationStrategy html, String llm, 
                           GenerationStrategy markdown, CroppingStrategy cropImage) {
        this.html = html;
        this.llm = llm;
        this.markdown = markdown;
        this.cropImage = cropImage;
    }

    // Getters and Setters
    public GenerationStrategy getHtml() {
        return html;
    }

    public void setHtml(GenerationStrategy html) {
        this.html = html;
    }

    public String getLlm() {
        return llm;
    }

    public void setLlm(String llm) {
        this.llm = llm;
    }

    public GenerationStrategy getMarkdown() {
        return markdown;
    }

    public void setMarkdown(GenerationStrategy markdown) {
        this.markdown = markdown;
    }

    public CroppingStrategy getCropImage() {
        return cropImage;
    }

    public void setCropImage(CroppingStrategy cropImage) {
        this.cropImage = cropImage;
    }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private GenerationStrategy html;
        private String llm;
        private GenerationStrategy markdown;
        private CroppingStrategy cropImage;

        public Builder html(GenerationStrategy html) {
            this.html = html;
            return this;
        }

        public Builder llm(String llm) {
            this.llm = llm;
            return this;
        }

        public Builder markdown(GenerationStrategy markdown) {
            this.markdown = markdown;
            return this;
        }

        public Builder cropImage(CroppingStrategy cropImage) {
            this.cropImage = cropImage;
            return this;
        }

        public GenerationConfig build() {
            return new GenerationConfig(html, llm, markdown, cropImage);
        }
    }
}