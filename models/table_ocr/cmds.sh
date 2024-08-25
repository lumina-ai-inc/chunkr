curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/akhileshsharma/Documents/Lumina/backend/api/models/table_ocr/sample/sample.png" \
  http://34.54.176.144/process-table


curl http://34.54.176.144/health

curl http://0.0.0.0:8000/health

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/akhileshsharma/Documents/Lumina/backend/api/models/table_ocr/sample/Screenshot 2024-08-21 at 9.42.22 PM.png" \
  -o sample.json \
  http://0.0.0.0:8000/ocr/table

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/akhileshsharma/Documents/Lumina/backend/api/models/table_ocr/sample/Screenshot 2024-08-21 at 9.42.22 PM.png" \
  -o sample.html \
  http://0.0.0.0:8000/html/table


curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/akhileshsharma/Documents/Lumina/backend/api/models/table_ocr/sample/sample.png" \
  -F "ocr_model=easyocr" \
  -o sample.html \
  http://0.0.0.0:8000/html/table

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/akhileshsharma/Documents/Lumina/backend/api/models/table_ocr/sample/Screenshot 2024-08-21 at 9.42.22 PM.png" \
  -F "ocr_model=easyocr" \
  -o sample.html \
  http://34.54.176.144/html/table

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/akhileshsharma/Documents/Lumina/backend/api/models/table_ocr/sample/sample.png" \
  -F "ocr_model=easyocr" \
  -o sample.json \
  http://0.0.0.0:8000/ocr/table


curl -F "options={\"languages\":[\"eng\"]}" \
  -F file=@/Users/akhileshsharma/Documents/Lumina/backend/api/models/table_ocr/sample/sample.png \
  -o ocr.json \
  http://0.0.0.0:8884/tesseract
