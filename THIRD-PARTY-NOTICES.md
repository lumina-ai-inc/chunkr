# THIRD-PARTY-NOTICES

This project uses third-party libraries that may be distributed under licenses different from the project's license. In the event that we accidentally failed to list a required notice, please bring it to our attention by raising an issue or reaching out in the Discord community.

The attached notices are provided for information only.

## pdf2image and PyPDF2

Copyright (c) 2006-2008, Mathieu Fenniak
Some contributions copyright (c) 2007, Ashish Kulkarni <kulkarni.ashish@gmail.com>
Some contributions copyright (c) 2014, Steve Witham <switham_github@mac-guyver.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* The name of the author may not be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


## pdf-document-layout-analysis

#### Note: We are currently using the apache-2.0 liscenced [pdf-document-layout-analysis](https://github.com/huridocs/pdf-document-layout-analysis) library as a part of our services. This helps us run inference code for VGT. We've made a few key changes to greatly improve performance on large page count pdfs. 

A copy of their original license is available [here](./pdf-document-layout-analysis/LICENSE) and in our services/pdf-document-layout-analysis directory.

This project includes modifications to the original source code located in the services/pdf-document-layout-analysis directory. The key changes are:

1. In src/data_model/PdfImages.py:
   - Added 'density' and 'extension' parameters to the from_pdf_path method.
   - Updated convert_from_path function to use 'density' parameter and set output format to 'jpeg'.
   - Modified pdf_name assignment logic.

2. In src/app.py:
   - Refactored routes for improved API convenience.
   - Added a 'density' parameter to the '/analyze/high-quality' route.
   - Implemented run_in_threadpool for the '/analyze/fast' route to enhance performance.
   - Added a processing lock for the '/analyze/high-quality' route to prevent concurrent processing.

3. API Changes:
   - Added new routes: '/readiness', '/', '/analyze/fast', '/analyze/high-quality'.
   - Removed routes: '/save_xml/{xml_file_name}', '/get_xml/{xml_file_name}', '/toc', '/text', '/error', and the general '/' POST route.

These modifications enhance flexibility in image conversion settings, improve API functionality and performance, and streamline the route structure.





