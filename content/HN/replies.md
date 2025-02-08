
Points:

- rd bench is outdated and was run when our server was on extreme load - doesn't replicate if you do it now. 
- cost comparison not for our cheapest tier - also we do a lot more than page -> llm. we actually solve the "trust" problem he had in his report.
- you can self host us and do 4pg/sec -> 11,000,000 pages per mo on consumer hardware (4090) which you can rent for 180 a month (0.25 cents per hr on vast.ai) + llm costs (((11,000,000)/1000)*0.65)=7150 dollars assuming you hit gemini pro 1.5 for tables (assuming one table per page which in practice is not true for most docs). AND you get INSIDE table bounding boxes.

- We can double down on layout analysis being the biggest break here - how if that works your downstream can be optimized and configured in numerous ways to achieve what you want

- simply can say we use gemini models - and the problems we've seen with flash vs pro

- Yeah we should just say we use Gemini for tables - the bench is very misrepresentative for x,y,z reason ?

- Main thing is we can go down the list for reply - man’s is right about DLA and bounding boxes being essential - then talk about downstream for OCR text being much more economical, faster and just all in all the reasonable approach for text - touch on how vlms suck full page and should only be used for specific complex processing for same speed and cost reasons - then all the last mile shit that also needs to be taken into consideration to give all the data needed for amazing rag