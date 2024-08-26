# Lumina Ingestion API

This README provides an overview of the Lumina Ingestion API, which allows you to process and convert PDFs into structured XML.

## API Endpoint

The base URL for the Lumina Ingestion API is:
`https://ingest.lumina.sh`


## Authentication

All API requests require an API key for authentication. Include your API key in the `x-api-key` header:

`x-api-key: your_api_key_here`

## Uploading Files

To upload a file for processing, send a POST request to the `/task` endpoint. We can currently process up to 100MB of PDFs in one request. For some context - we're able to use this system to process up to 100 files(~20 pages/file) per second. 

### Request

- **Method**: POST
- **URL**: `https://ingest.lumina.sh/task`
- **Headers**:
  - `x-api-key: your_api_key_here`
- **Body**: Form-data with the file to be processed

### Example Request

To upload a file named `sample.pdf` located in the `./examples` directory:

Use the following curl command:

    curl -v --request POST \
    --url https://ingest.lumina.sh/task \
    -H "x-api-key: your_api_key_here" \
    -F file=@./examples/{file1}.pdf
    -F file=@./examples/{file2}.pdf
    -F file=@./examples/{file3}.pdf
    -F file=@./examples/{file4}.pdf
    ...

### Response

The API will respond with a JSON object containing details about the created task:

```json
{
  "created_at": "2024-01-01T00:00:00.000000Z",
  "expires_at": "2024-01-01T01:00:00.000000Z",
  "files": [
    {
      "id": "file-id-1",
      "status": "succeeded",
      "input_url": "https://example.com/filename1.pdf",
      "output_url": "https://example.com/filename1.xml"
    }
  ],
  "finished_at": "2024-01-01T00:00:10.000000Z",
  "message": "",
  "status": "succeeded",
  "task_id": "{task_id}",
  "url": "https://ingest.lumina.sh/task/{task_id}"
}
```

## Checking Task Status

You can check the status of a task by sending a GET request to the `/task/{task_id}` endpoint.

### Request

- **Method**: GET
- **URL**: `https://ingest.lumina.sh/task/{task_id}`
- **Headers**:
  - `x-api-key: your_api_key_here`

### Response

The response will be similar to the one received when creating a task, but with updated status information.

## Task Statuses

A task can have one of the following statuses:

- `starting`: The task is being initialized
- `processing`: The task is currently processing the file(s)
- `succeeded`: All files in the task have been successfully processed
- `failed`: The task encountered an error during processing

## File Statuses

A file can have one of the following statuses:

- `starting`: The file is ready to be processed
- `processing`: The file is currently being processed
- `succeeded`: The file has been successfully processed
- `failed`: The file encountered an error during processing

## File URLs

The `input_url` and `output_url` provided in the response are pre-signed URLs that are valid for 10 minutes. These URLs allow secure access to the original and processed files, respectively.

The `input_url` is the original file and the `output_url` is the processed file

## Task Expiration

Tasks expire after 1 hour. Make sure to retrieve any necessary information or download processed files before the expiration time.

## Error Handling

If an error occurs during processing, the `status` field will be set to `failed`, and the `message` field will contain more information about the error.

# Lumina Ingestion API Output

Each file processed by the Lumina Ingestion API is converted to TEI/XML format. The TEI/XML format is a standardized format for encoding text and data in XML.


## Annotation guidelines for the TEI/XML results

The lumina api attempts to recognize the following objects:

* title
* paragraphs
* section titles
* figures
* tables
* formulas
* list items inside lists
* markers: callouts to figures ("see Fig. 1"), tables, formulas and to bibliographical references (e.g. "Austin 2008(b)").

### Title

The title of the file is encoded under the TEI element `<title>` with attributes `level="a"` and `type="main"`.

```xml
<title level="a" type="main" coord="1,51.02,111.26,416.69,14.54;1,51.01,129.17,419.42,14.54;1,51.01,147.14,367.68,14.54">Opposite environmental gating of the experienced utility (&apos;liking&apos;) and decision utility (&apos;wanting&apos;) of heroin versus cocaine in animals and humans: implications for computational neuroscience</title>
```

### Paragraphs

Paragraphs contain text which in turn may contain inline elements such as references (see below) or line breaks.

```xml
<p>Our group has investigated the correlation between sphingolipid metabolism, the<lb/>
  ...
  are able to induce autophagy in a breast cancer cell line. 3,4<lb/>
</p>
```

> Note: The `<lb/>` (line break) elements are there because they have been recognized as such in the PDF in the text flow. However the fact that they are located within or outside a tagged paragraph or section title has no impact. 

Following the TEI, formulas should be on the same hierarchical level as paragraphs, and not be contained inside paragraphs:

```xml
<p>Exponentiation mixes. Our protocol will benefit from the exponentiation mix<lb/>
	...
  MS i+1 . The first server takes the original list of PKs. The net effect is a list: <lb/>
</p>

<formula>(g x ρ(i) ·s ) i∈[n] ,<lb/> </formula>

<p>where:<lb/></p>

<formula>π =<lb/>
  [n]<lb/>
  1<lb/>
  ρ i , s =<lb/>
  [n]<lb/>
  1<lb/>
  s i ,<lb/>
</formula>

<p>and g s is also published by the last Teller.<lb/>
  ...
  (g s ) xi and finding the match.<lb/>
</p>
```

The next example illustrates similarly that in TEI list items should be contained inside `<list>` elements which in turn are on the same hierarchical level as paragraphs and other block-level elements.

```xml
<p>The estimation of the eligible own funds and the SCR requires to carry out calculations <lb/>
  ...
  the following constraints:<lb/>
</p>

<list>
  <item>• updating the assets and liabilities model points;<lb/></item>

  <item>• constructing a set of economic scenarios under the risk-neutral probability and<lb/>
	  checking its market-consistency;<lb/></item>
</list>

<p>The wild-type strain was Bristol N2. All animals were raised at<lb/>
  20uC. The following alleles and transgenes were used:<lb/>
</p>

<p>LGI: hda-3(ok1991)<lb/>
</p>

<p>LGII: hda-2(ok1479) <lb/>
</p>
```

### Section titles

The titles of sections are encoded under the TEI element `<head>`.

```xml
<head>CERAMIDE AND S1P BOTH TRIGGER AUTOPHAGY<lb/></head>
```

```xml
<head>1. Introduction<lb/></head>
<head>2 Background<lb/></head>
```

```xml
<head>B. Focusing an IACT<lb/></head>
```

```xml
<head>II. PROBLEM AND SOLUTION PROCEDURE<lb/></head>
```

```xml
<head>4 RESULTS<lb/></head>
<head>4.1 Image quality<lb/></head>
```

```xml
<head>Results<lb/></head>
<head>Patient characteristics<lb/></head>
```

```xml
<head>MATERIALS AND METHODS<lb/></head>
<head>Tissue samples<lb/></head>
```

### Figures, tables and box

A photo, picture or other graphical representation (this could be a chart or another figure) and boxes, will be encoded under the TEI element `<figure>`. This element contains the title, the figure/table/boxed content/photo itself, captions, any legend or notes it may have.

Note that following the TEI, a table is encoded as a figure of type "table" (the actual `<table>` element appears in the `table` model applied in cascade) and a boxed content is marked as a figure of type "box".

```xml
<figure>Figure 1. Hypothetical model for ceramide and S1P-induced autophagy and thei	consequences on cell fate. An<lb/>
  ...
  <lb/>
  ....
  promotes cell survival by inhibiting the induction of apoptosis.<lb/>
</figure>

<figure type="table"> Table 1 Clades of clownfishes used in this study<lb/>
  Clade name<lb/>
	Species<lb/>
  percula<lb/>
  A. ocellaris, A. percula, P. biaculeatus<lb/>
  Australian<lb/>
	A. akindynos, A. mccullochi<lb/>
  ...
  of clownfish species [19].<lb/>
</figure>

<figure type="table"> Table 1 The clinicopathological data of PDAC tissue samples<lb/>
  Sample<lb/>
  Age<lb/>
  Sex<lb/>
  Location a<lb/>
  Histology b<lb/>
  T<lb/>
  N<lb/>
  ...
  1<lb/>
  1<lb/>
  IVb<lb/>
  a<lb/>
  P ¼ primary lesion; Ph ¼ head; Pb ¼ body; Pt ¼ tail of the pancreas; LM ¼ liver metastatic lesion. b
	mod ¼ moderately; poor ¼ poorly differentiated tubular adenocarcinoma.<lb/>
  PDAC ¼ pancreatic ductal adenocarcinoma; FISH ¼ fluorescence in situ hybridisation; ISH ¼ in
	situ RNA hybridisation.<lb/>
</figure>
```

Boxed content, i.e. a box with additional content __outside__ the flow of the general content, are tagged similarly with the element `<figure type="box">`.

### Formulas


The `<formula>` tag is used to identify a formula appearing as an independent block in the text body. This formula often comes with a label, its "reference marker", which can be used for callout to the formula in the text body. Shorter inline formulas are not specifically annotated, they are considered as part of the body.  

The label of a formula is usually a number, but can be any symbols. It is tagged with the element `<label>`, as illustrated bellow:


```xml
<formula>
  σ α β =<lb/>
  1<lb/>
  3<lb/>
  ˙<lb/>
  R<lb/>
  R −<lb/>
  ˙<lb/>
  R<lb/>
  R<lb/>
  diag(0, 2, −1, −1),<lb/>
	(<label>10</label>)
  <lb/>
</formula>
```

### List items

Following the TEI, list items (`<item>` elements) should be contained in a `<list>` element and must not occur within `<p>` elements. At this stage no difference is made between ordered and unordered lists.

List item markers such as hyphens, bullet points (for unordered lists) or numbers and letters (for ordered lists) should be contained within the `<item>` element.

```xml
<p>Introducing ballot identifiers has the appeal that it provides voters with a<lb/>
  very simple, direct and easy to understand way to confirm that their vote is<lb/>
  ...
  this observation that we exploit to counter this threat: we arrange for the voters<lb/>
  to learn their tracker numbers only after the information has been posted to the<lb/>
  WBB.<lb/>
  This paper presents a scheme that addresses both of these shortcomings by:<lb/>
</p>

<list>
  <item>– Guaranteeing that voters get unique trackers.<lb/></item>

  <item>– Arranging for voters to learn their tracker only after the votes and corre-<lb/>
  sponding tracking numbers have been posted (in the clear).<lb/></item>
</list>

<list>
	<item>1) The difficulty of identifying passages in a user&apos;s manual based on an individual word.<lb/></item>

  <item>2) The difficulty of distinguishing affirmative and negative sentences which mean	two different<lb/>
  features in the manual.<lb/></item>

  <item>3) The difficulty of retrieving appropriate passages for a query using words not appearing in the<lb/>
  manual.<lb/></item>
</list>
```

### Markers (callouts to structures)

These elements appear as inline elements, inside `<p>`, `<item>`, or other elements containing and usual reference other parts of the document. They could be understood as links.  Here is a list of currently supported markers:

* `<ref type="biblio">` a link to a bibliographical reference (of the type *see **Austin 1982/b** * )
* `<ref type="figure">` a pointer to a figure elsewhere in the document (*Fig 5b, left*)
* `<ref type="table">` a link to a table in the document
* `<ref type="box">` a link to some boxed content (a box with additional content, outside the flow of the general content)
* `<ref type="formula">` a link to a formula

#### Bibliographical reference markers (callouts)

The following excerpts show examples of markers (callouts) to bibliographical references introduced with numbers with or without brackets or parenthesis:

```xml
Harbaugh and Harbaugh <ref type="biblio">[7]</ref>
```

```xml
in Lolle et al. <ref type="biblio">1</ref>
```

Note that the bracket/parenthesis symbols are **included** in the tagged content.

Bellow, as the label within the brackets fully qualifies the reference, we don't further annotate the callout with the author names:

```xml
<p>The clinical entity of cervical flexion myelopa-<lb/>
  thy proposed in the 1960s by Reid <ref type="biblio">[16,17]</ref> and<lb/>
  Breig et al <ref type="biblio">[2,3]</ref> has been neglected for a long time.<lb/>
  ...
  groups of males and females controls.<lb/>
</p>
```

It is important to note that whenever there is an enumeration of several references, they should not be marked up as several references, using specific `ref` elements for each one, but just one element surrounding the whole group. 

To highlight the diversity of bibliographical references, here are some more examples:

```xml
by <ref type="biblio">Greve et al. [1994]</ref> and <ref type="biblio">Koch et al.<lb/> [1994]</ref>
```

```xml
<ref type="biblio">[Whitham, 1954; Hogg and Pritchard,<lb/> 2004]</ref>
```

```xml
<ref type="biblio">[Abramowitz and Stegun,<lb/> 1964, see pp. 559 – 562] </ref>
```

```xml
<ref type="biblio">LEWANDOWSKI (1982b)</ref> has reported
```

```xml
<ref type="biblio">(STANCZYKOWSKA, 1977)</ref>
```

```xml
(e.g., <ref type="biblio">Bryant<lb/> &amp; Goswami, 1987</ref>) by<lb/>
<ref type="biblio">Smith and Matsoukas [1998]</ref> and <ref type="biblio">Khelifa and Hill [2006b]<lb/></ref>
```

```xml
by <ref type="biblio">Gal [1979]</ref> and<lb/> illustrated further recently by <ref type="biblio">Draper and Lund [2004]</ref>
```

```xml
<ref type="biblio">(ref. 1)</ref>
```

Note that strange style mixtures can be observed:

```xml
This was established in<lb/> <ref type="biblio">Thompson v. Lochert (1997)</ref> <ref type="biblio">[114]</ref> in which
```

```xml
such as receptors for IL-1β <ref type="biblio">(REFS 64,65)</ref>,<lb/> TNF <ref type="biblio">66</ref> and IFNγ <ref type="biblio">67</ref> .
```

#### Markers to tables, figures and formula

The next example shows markers (callouts) to a table and a figure (as noted earlier, whitespace is not of importance and can therefore be used liberally, like here to better show the tagging):

```xml
<p>The patient group comprised all six patients with<lb/>
  juvenile cervical
	flexion myelopathy admitted to<lb/>
  our hospital

   (Table <ref type="table">1</ref>).

  In all of them, cervical flexion<lb/>
  ...
  alignment in the extended neck position

  (Figure <ref type="figure">3</ref>).<lb/>

  Cervical MR imaging in the neutral neck position of<lb/>
  five of the six patients showed a straight cervical<lb/>
</p>
```

An example of callouts to two formulas (and a bibliographical entry):

```xml
<p>Here, Θ(y) denotes ...<lb/>
the semi-cylindrical drum. The dynamics of the avalanches of eqs.

  (<ref type="formula">1</ref>)

  and

  (<ref type="formula">2</ref>)

  is centered<lb/>
  around the angle ϕ d = tan b 0

  <ref type="biblio">[7]</ref>.<lb/>

  ...
  stochastic extension of the DMM:<lb/>
</p>
```

As visible in the examples, markers to figure, table or formula are annotated by including only the key information of the refered object. Brackets, parenthesis, extra wording, extra information are left outside of the tagged text (in contrast to bibliographical markers, where we keep the brackets and parenthesis by convention). 
Here are some more short examples for figure markers:

```xml
(Supplementary Fig. <ref type="figure">1</ref><lb/> online)
```

```xml
(Fig. <ref type="figure">5b</ref>, left)
```

```xml
Figure <ref type="figure">2</ref> exemplifies
```

```xml
(10.3% of those analysed; Fig. <ref type="figure">1a</ref>).
```

As for markers to bibliographical references, we group under the same element a conjunction/list of callouts:

```xml
In figs. <ref type="figure">3 and 4</ref>
```

## Coordinates in TEI/XML results

The following elements have coords:
* `title`
* `persName`
* `figure`
* `ref`
* `biblStruct`
* `formula`
* `p`
* `s`
* `note`
* `head`

## Coordinates in TEI/XML results

A __bounding box__ is defined by the following attributes: 

- `p`: the number of the page (beware, in the PDF world the first page has index 1!), 

- `x`: the x-axis coordinate of the upper-left point of the bounding box,

- `y`: the y-axis coordinate of the upper-left point of the bounding box (beware, in the PDF world the y-axis extends downward!),

- `h`: the height of the bounding box,

- `w`: the width of the bounding box.

Coordinates for a given structure appear via an extra attribute ```@coords```. 

* The list of page size is encoded under the TEI element `<facsimile>`. The dimension of each page is given successively by the TEI attributes `@lrx` and `@lry` of the element `<surface>` to be conformant with the TEI (`@ulx` and `@uly` are used to set the orgine coordinates, which is always `(0,0)` for us). **This is generally not needed to create bounding boxes.**

Example: 


```xml
	...
	</teiHeader>
	<facsimile>
		<surface n="1" ulx="0.0" uly="0.0" lrx="612.0" lry="794.0"/>
		<surface n="2" ulx="0.0" uly="0.0" lrx="612.0" lry="794.0"/>
		<surface n="3" ulx="0.0" uly="0.0" lrx="612.0" lry="794.0"/>
		<surface n="4" ulx="0.0" uly="0.0" lrx="612.0" lry="794.0"/>
		<surface n="5" ulx="0.0" uly="0.0" lrx="612.0" lry="794.0"/>
	</facsimile>
	<text xml:lang="en">
	...
```

* The coordinates of a structure is provided as a list of bounding boxes, each one separated by a semicolon ```;```, each bounding box being defined by 5 attributes separated by a comma ```,```:

Example 1: 
```xml
<author>
	<persName coords="1,53.80,194.57,58.71,9.29">
		<forename type="first">Ron</forename>
		<forename type="middle">J</forename>
		<surname>Keizer</surname>
	</persName>
</author>
```

"1,53.80,194.57,58.71,9.29" indicates one bounding box with attributes page=1, x=53.80, y=194.57, w=58.71, h=9.29.

Example 2:
```xml
<biblStruct coords="10,317.03,183.61,223.16,7.55;10,317.03,192.57,223.21,7.55;10,317.03,201.53,223.15,7.55;10,317.03,210.49,52.22,7.55"  xml:id="b19">
```

The above ```@coords``` XML attributes introduces 4 bounding boxes to define the area of the bibliographical reference (typically because the reference is on several line).

As side note, in traditionnal TEI encoding an area should be expressed using SVG. However it would have make the TEI document quickly unreadable and extremely heavy and we are using this more compact notation. 

Here is a sample python code to overlay the bounding boxes on the PDF:

```python
pip install pymupdf
```

```python
import fitz 
def overlay_bounding_boxes(pdf_path, file_coords, output_dir):
    doc = fitz.open(pdf_path)
    for coords in file_coords:
        coords_set = [coord for coord in coords.split(';') if coord]
        
        if coords_set:
            # Initialize variables to store the extremes of the bounding box
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            page_num = None

            for box in coords_set:
                try:
                    p, x, y, w, h = map(float, box.split(','))
                    if page_num is None:
                        page_num = int(p) - 1
                    
                    # Update the extremes
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)
                except Exception as e:
                    print(f"Error processing box {box}: {e}")

            if page_num is not None:
                page = doc[page_num]
                rect = fitz.Rect(min_x, min_y, max_x, max_y)
                page.draw_rect(rect, color=(1, 0, 0), width=1)

    output_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}")
    try:
        doc.save(output_path)
    except Exception as e:
        print(f"Error saving PDF to {output_path}: {e}")
    finally:
        doc.close()
```



## Support

For any questions or issues, please contact our support team at akhilesh@lumina.sh.