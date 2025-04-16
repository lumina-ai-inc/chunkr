# TaskResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**configuration** | [**models::Configuration**](Configuration.md) |  | 
**created_at** | **String** | The date and time when the task was created and queued. | 
**expires_at** | Option<**String**> | The date and time when the task will expire. | [optional]
**finished_at** | Option<**String**> | The date and time when the task was finished. | [optional]
**message** | **String** | A message describing the task's status or any errors that occurred. | 
**output** | Option<[**models::OutputResponse**](OutputResponse.md)> |  | [optional]
**started_at** | Option<**String**> | The date and time when the task was started. | [optional]
**status** | [**models::Status**](Status.md) |  | 
**task_id** | **String** | The unique identifier for the task. | 
**task_url** | Option<**String**> | The presigned URL of the task. | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


