# Segment

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**bbox** | [**models::BoundingBox**](BoundingBox.md) |  | 
**confidence** | Option<**f32**> | Confidence score of the layout analysis model | [optional]
**content** | Option<**String**> | Text content of the segment. Calculated by the OCR results. | [optional]
**html** | Option<**String**> | HTML representation of the segment. | [optional]
**image** | Option<**String**> | Presigned URL to the image of the segment. | [optional]
**llm** | Option<**String**> | LLM representation of the segment. | [optional]
**markdown** | Option<**String**> | Markdown representation of the segment. | [optional]
**ocr** | Option<[**Vec<models::OcrResult>**](OCRResult.md)> | OCR results for the segment. | [optional]
**page_height** | **f32** | Height of the page containing the segment. | 
**page_number** | **i32** | Page number of the segment. | 
**page_width** | **f32** | Width of the page containing the segment. | 
**segment_id** | **String** | Unique identifier for the segment. | 
**segment_type** | [**models::SegmentType**](SegmentType.md) |  | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


