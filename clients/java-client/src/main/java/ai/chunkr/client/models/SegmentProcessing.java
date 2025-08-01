package ai.chunkr.client.models;



/**
 * Configuration for processing different segment types.
 */

public class SegmentProcessing {
    

    private GenerationConfig title;
    

    private GenerationConfig sectionHeader;
    

    private GenerationConfig text;
    

    private GenerationConfig listItem;
    

    private GenerationConfig table;
    

    private GenerationConfig picture;
    

    private GenerationConfig caption;
    

    private GenerationConfig formula;
    

    private GenerationConfig footnote;
    

    private GenerationConfig pageHeader;
    

    private GenerationConfig pageFooter;
    

    private GenerationConfig page;

    // Constructors
    public SegmentProcessing() {}

    // Getters and Setters
    public GenerationConfig getTitle() {
        return title;
    }

    public void setTitle(GenerationConfig title) {
        this.title = title;
    }

    public GenerationConfig getSectionHeader() {
        return sectionHeader;
    }

    public void setSectionHeader(GenerationConfig sectionHeader) {
        this.sectionHeader = sectionHeader;
    }

    public GenerationConfig getText() {
        return text;
    }

    public void setText(GenerationConfig text) {
        this.text = text;
    }

    public GenerationConfig getListItem() {
        return listItem;
    }

    public void setListItem(GenerationConfig listItem) {
        this.listItem = listItem;
    }

    public GenerationConfig getTable() {
        return table;
    }

    public void setTable(GenerationConfig table) {
        this.table = table;
    }

    public GenerationConfig getPicture() {
        return picture;
    }

    public void setPicture(GenerationConfig picture) {
        this.picture = picture;
    }

    public GenerationConfig getCaption() {
        return caption;
    }

    public void setCaption(GenerationConfig caption) {
        this.caption = caption;
    }

    public GenerationConfig getFormula() {
        return formula;
    }

    public void setFormula(GenerationConfig formula) {
        this.formula = formula;
    }

    public GenerationConfig getFootnote() {
        return footnote;
    }

    public void setFootnote(GenerationConfig footnote) {
        this.footnote = footnote;
    }

    public GenerationConfig getPageHeader() {
        return pageHeader;
    }

    public void setPageHeader(GenerationConfig pageHeader) {
        this.pageHeader = pageHeader;
    }

    public GenerationConfig getPageFooter() {
        return pageFooter;
    }

    public void setPageFooter(GenerationConfig pageFooter) {
        this.pageFooter = pageFooter;
    }

    public GenerationConfig getPage() {
        return page;
    }

    public void setPage(GenerationConfig page) {
        this.page = page;
    }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private GenerationConfig title;
        private GenerationConfig sectionHeader;
        private GenerationConfig text;
        private GenerationConfig listItem;
        private GenerationConfig table;
        private GenerationConfig picture;
        private GenerationConfig caption;
        private GenerationConfig formula;
        private GenerationConfig footnote;
        private GenerationConfig pageHeader;
        private GenerationConfig pageFooter;
        private GenerationConfig page;

        public Builder title(GenerationConfig title) {
            this.title = title;
            return this;
        }

        public Builder sectionHeader(GenerationConfig sectionHeader) {
            this.sectionHeader = sectionHeader;
            return this;
        }

        public Builder text(GenerationConfig text) {
            this.text = text;
            return this;
        }

        public Builder listItem(GenerationConfig listItem) {
            this.listItem = listItem;
            return this;
        }

        public Builder table(GenerationConfig table) {
            this.table = table;
            return this;
        }

        public Builder picture(GenerationConfig picture) {
            this.picture = picture;
            return this;
        }

        public Builder caption(GenerationConfig caption) {
            this.caption = caption;
            return this;
        }

        public Builder formula(GenerationConfig formula) {
            this.formula = formula;
            return this;
        }

        public Builder footnote(GenerationConfig footnote) {
            this.footnote = footnote;
            return this;
        }

        public Builder pageHeader(GenerationConfig pageHeader) {
            this.pageHeader = pageHeader;
            return this;
        }

        public Builder pageFooter(GenerationConfig pageFooter) {
            this.pageFooter = pageFooter;
            return this;
        }

        public Builder page(GenerationConfig page) {
            this.page = page;
            return this;
        }

        public SegmentProcessing build() {
            SegmentProcessing segmentProcessing = new SegmentProcessing();
            segmentProcessing.title = title;
            segmentProcessing.sectionHeader = sectionHeader;
            segmentProcessing.text = text;
            segmentProcessing.listItem = listItem;
            segmentProcessing.table = table;
            segmentProcessing.picture = picture;
            segmentProcessing.caption = caption;
            segmentProcessing.formula = formula;
            segmentProcessing.footnote = footnote;
            segmentProcessing.pageHeader = pageHeader;
            segmentProcessing.pageFooter = pageFooter;
            segmentProcessing.page = page;
            return segmentProcessing;
        }
    }
}