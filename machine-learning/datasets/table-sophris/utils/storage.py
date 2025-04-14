import boto3
import os
from pathlib import Path
import logging
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('table_storage')

class TableS3Storage:

    def __init__(self, s3_bucket=None, dataset_name=None, max_workers=10):
        self.s3_bucket = s3_bucket or os.environ.get("S3_BUCKET", "chunkr-datasets")
        self.dataset_name = dataset_name or os.environ.get("DATASET_NAME", "azure-sophris-distill")
        self.max_workers = max_workers
        self.temp_dir = Path("temp_table_downloads")
        self.temp_dir.mkdir(exist_ok=True)
        
        self.base_prefix = ""
        
        self.transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            max_concurrency=max_workers,
            multipart_chunksize=8 * 1024 * 1024,
            use_threads=True
        )

        self.s3_client = None
        try:
            self.s3_client = boto3.client('s3')
            logger.info(f"S3 interface initialized with bucket: {self.s3_bucket}, dataset: {self.dataset_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            self.s3_client = None

    def is_enabled(self):
        return self.s3_client is not None

    def _get_dataset_prefix(self, dataset_name=None):
        dataset = dataset_name or self.dataset_name
        return f"{self.base_prefix}{dataset}/"

    def _get_tables_prefix(self, dataset_name=None):
        return f"{self._get_dataset_prefix(dataset_name)}tables/"

    def _get_groundtruth_prefix(self, dataset_name=None):
        return f"{self._get_dataset_prefix(dataset_name)}groundtruth/"
    
    def _get_table_key(self, table_name, dataset_name=None):
        return f"{self._get_tables_prefix(dataset_name)}{table_name}.html"
    
    def _get_groundtruth_key(self, table_name, dataset_name=None):
        return f"{self._get_groundtruth_prefix(dataset_name)}{table_name}.html"

    def list_datasets(self):
        if not self.is_enabled():
            return []

        datasets = set()
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.base_prefix, Delimiter='/'):
                for prefix_info in page.get('CommonPrefixes', []):
                    full_prefix = prefix_info.get('Prefix', '')
                    dataset_id = full_prefix[len(self.base_prefix):].strip('/')
                    if dataset_id:
                        datasets.add(dataset_id)
            logger.info(f"Found {len(datasets)} table datasets: {list(datasets)}")
            return sorted(list(datasets))
        except Exception as e:
            logger.error(f"Error listing table datasets: {str(e)}")
            return []

    def list_tables(self, dataset_name=None):
        if not self.is_enabled():
            return []

        tables = []
        tables_prefix = self._get_tables_prefix(dataset_name)
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=tables_prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.html'):
                        filename = key.split('/')[-1]
                        table_name = filename[:-5]
                        tables.append(table_name)
            
            logger.info(f"Found {len(tables)} tables in dataset '{dataset_name or self.dataset_name}'")
            return sorted(tables)
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []

    def upload_table_html(self, table_name, html_content, dataset_name=None):
        if not self.is_enabled():
            return False
            
        s3_key = self._get_table_key(table_name, dataset_name)
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=html_content.encode('utf-8'),
                ContentType='text/html'
            )
            logger.info(f"Uploaded table HTML: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading table HTML {s3_key}: {str(e)}")
            return False

    def upload_groundtruth_html(self, table_name, html_content, dataset_name=None):
        if not self.is_enabled():
            return False
            
        s3_key = self._get_groundtruth_key(table_name, dataset_name)
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=html_content.encode('utf-8'),
                ContentType='text/html'
            )
            logger.info(f"Uploaded ground truth HTML: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading ground truth HTML {s3_key}: {str(e)}")
            return False

    def download_table_html(self, table_name, dataset_name=None):
        if not self.is_enabled():
            return None
            
        s3_key = self._get_table_key(table_name, dataset_name)
        
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            logger.info(f"Downloaded table HTML: {s3_key}")
            return content
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Table HTML not found: {s3_key}")
            else:
                logger.error(f"Error downloading table HTML {s3_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading table HTML {s3_key}: {str(e)}")
            return None

    def download_groundtruth_html(self, table_name, dataset_name=None):
        if not self.is_enabled():
            return None
            
        s3_key = self._get_groundtruth_key(table_name, dataset_name)
        
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            logger.info(f"Downloaded ground truth HTML: {s3_key}")
            return content
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Ground truth HTML not found: {s3_key}")
            else:
                logger.error(f"Error downloading ground truth HTML {s3_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading ground truth HTML {s3_key}: {str(e)}")
            return None

    def ensure_dataset_structure(self, dataset_name=None):
        if not self.is_enabled():
            return False
            
        dataset = dataset_name or self.dataset_name
        
        prefixes_to_ensure = [
            self._get_dataset_prefix(dataset),
            self._get_tables_prefix(dataset),
            self._get_groundtruth_prefix(dataset)
        ]
        
        try:
            for prefix in prefixes_to_ensure:
                try:
                    self.s3_client.head_object(Bucket=self.s3_bucket, Key=prefix)
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        self.s3_client.put_object(Bucket=self.s3_bucket, Key=prefix, Body='')
                        logger.info(f"Created S3 folder placeholder: {prefix}")
                    else:
                        raise
            
            logger.info(f"Ensured dataset structure for: {dataset}")
            return True
        except Exception as e:
            logger.error(f"Error ensuring dataset structure: {str(e)}")
            return False

    def upload_table_with_groundtruth(self, table_name, table_html, groundtruth_html, dataset_name=None):
        self.ensure_dataset_structure(dataset_name)
        
        table_success = self.upload_table_html(table_name, table_html, dataset_name)
        groundtruth_success = self.upload_groundtruth_html(table_name, groundtruth_html, dataset_name)
        
        return table_success and groundtruth_success

    def delete_table(self, table_name, dataset_name=None):
        if not self.is_enabled():
            return False
            
        dataset = dataset_name or self.dataset_name
        table_key = self._get_table_key(table_name, dataset)
        groundtruth_key = self._get_groundtruth_key(table_name, dataset)
        
        try:
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=table_key)
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=groundtruth_key)
            
            logger.info(f"Deleted table and ground truth for: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting table {table_name}: {str(e)}")
            return False

    def cleanup_temp_dir(self):
        try:
            for item in self.temp_dir.iterdir():
                if item.is_file():
                    item.unlink()
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")

def main():
    storage = TableS3Storage()
    
    if not storage.is_enabled():
        logger.error("S3 storage not properly initialized. Exiting.")
        return
    
    storage.ensure_dataset_structure()
    
    datasets = storage.list_datasets()
    logger.info(f"Available datasets: {datasets}")
    
    tables = storage.list_tables()
    logger.info(f"Tables in dataset: {tables}")
    
    sample_table_html = "<html><body><table><tr><td>Sample</td></tr></table></body></html>"
    sample_groundtruth_html = "<html><body><table><tr><td>Ground Truth</td></tr></table></body></html>"
    
    storage.upload_table_with_groundtruth(
        "sample_table", 
        sample_table_html, 
        sample_groundtruth_html
    )
    
    storage.cleanup_temp_dir()

if __name__ == "__main__":
    main()
