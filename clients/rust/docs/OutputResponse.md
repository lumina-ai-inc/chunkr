# OutputResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**chunks** | [**Vec<models::Chunk>**](Chunk.md) | Collection of document chunks, where each chunk contains one or more segments | 
**extracted_json** | Option<[**serde_json::Value**](.md)> |  | [optional]
**file_name** | Option<**String**> | The name of the file. | [optional]
**page_count** | Option<**i32**> | The number of pages in the file. | [optional]
**pdf_url** | Option<**String**> | The presigned URL of the PDF file. | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


