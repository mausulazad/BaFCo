from utils import LONG_FORM_LABEL_CATEGORIES, SHORT_FORM_LABEL_CATEGORIES


DLA_FULL_ZS_PROMPT = """You are a **form layout understanding AI**. Given an image of a form page, your task is to analyze its layout and identify all visible fields along with their categories and bounding boxes.
   input: 
   a. image of form page
   b. full image size (height and width)
   c. form field categories
   
   TASKS:
   1. Localize every visible field on the page.
   2. Classify each field into one of the given form field categories.
   3. Return bounding boxes [X, Y, W, H], where (X, Y) is the top-left corner, W is width, and H is height.
   
   OUTPUT FORMAT(exactly as shown):
   [
    {{
      "id": "<unique_id>",
      "reasoning": <brief 1-line non-generic reasoning on choice of class>,
      "class": "<category_name>",
      "bbox": <Length 4 list [X, Y, W, H] where (X, Y) represents left-upper corner of bounding box and W and H width and height of the box respectively>,
      "confidence": <float 0-1>
    }}
   ]

   COORDINATES:
   - Origin (0,0) = top-left; units = pixels.
   - Boxes tightly bound visible content, EXCEPT value fields, which should include the full writable area.
   - Overlap only when nested (e.g., key → value).
   - Coordinates must be absolute pixel values expressed as floats (at most 4 decimal places).
   - Do NOT use repeating decimal patterns.


   FIELD RULES:
   - **Header, Footer, Title of Form, Section Title, Page Num**: structural bands or labels.
   - **Form Key**: label requesting input (“Name”, “Date of Birth”).
   - **Form Value**: blank region for user entry; spacious bounding box.
   - **Inline Key, Inline Value**: key phrases and blanks within a sentence; value = writable area.
   - **Checkbox, Tick Mark**: selectable elements (“☐ Male”, “Yes / No”).
   - **Signature Key, Signature Val**: signature label and signing area (spacious box).
   - **Photo Field**: placeholders for attaching photos 
   - **Figure/Diagram/Logo**: icons. (This is one category covering figures, diagrams, and logos)
   - **Text Block**: instructions or narrative text.
   - **Gibberish, Mark for removal**: scribbles, stray marks.
   - **Others**: fallback for unclassified regions.

   TABLE RULES:
   A. STRUCTURE
   - Draw one **Table** box around the full grid region (even if lines are faint).
   - Choose one decomposition style:
    • **Column-major** → columns have headers at the top.  
    • **Row-major** → rows have labels on the left.  
    Do not mix both for one table.

   B. RELATIONSHIPS
   - Internal elements (keys, values, captions, section titles) set parent_id to the Table.
   - **Table Caption**: title above/below the table.
   - **Table Index**: short identifier referencing the table (e.g., “Table 1”, “Table II”, “Exhibit A”), usually placed immediately before the caption.
   - **Table Section Title**: header band dividing row/column groups.

   C. COLUMN-MAJOR
   - **Table Col PKey**: header cell areas across the top.
   - **Table Col Value**: vertical strips below headers spanning to table bottom.
   - Merged headers → one PKey linked to multiple Value regions
   - No overlap among adjacent columns.

   D. ROW-MAJOR
   - **Table Row PKey**: leftmost identifying cells for each row.
   - **Table Row Value**: remaining cells of that row.
   - Row Value spans full row height for multi-line content.

   E. MERGED / IRREGULAR CELLS
   - Assign each merged cell as key or value based on function.
   - Adjust boxes to include full merged area.

   F. INVISIBLE GRIDS
   - Infer alignment from consistent gutters or repeated leaders.
   - Keep column/row edges straight and globally aligned.

   G. CONSISTENCY
   - Keys = label regions; Values = data regions.
   - No overlaps among peers.
   - Preserve inner whitespace for writable cells.

   H. SECTION TITLES INSIDE TABLES
   - Bold or shaded header rows spanning multiple columns → Table Section Title.

   I. MULTI-PAGE TABLES
   - Annotate each page separately; repeat headers when they reappear.

   J. TEXT PREVIEW
   - Include short snippet if legible (“Name”, “Total”), else empty string.

   QUALITY RULES:
   - Detect all visible fields; no omissions.
   - Use visual layout, proximity, and language.
   - Prefer meaningful, tighly bounded boxes.
   - Do not output a single object unless the page truly contains only one visible element.
   - Do not stop after the first detection if more than one visible element exist. Keep iterating until you have enumerated every visible field.
   - Use “Others” for uncertain cases.
   - Output must be in described output format (a collection of JSON objects) — no other key, natural text or comments.
   
   Full Image Size: Height -> {form_image_height}, Width -> {form_image_width}
   Available Form Field Categories: {form_field_categories}
   Use only the category names exactly as provided in 'Available Form Field Categories'. Do not invent, abbreviate, or modify category names.
   Image: <image>
   Your Layout Analysis Output:"""


DLA_REDUCED_ZS_PROMPT = """You are a **form layout understanding AI**. Given an image of a form page, your task is to analyze its layout and identify all visible fields along with their categories and bounding boxes.
input:
a. image of form page
b. full image size (height and width)
c. form field categories

TASKS:
1. Localize every visible field on the page.
2. Classify each field into one of the given form field categories.
3. Return bounding boxes [X, Y, W, H], where (X, Y) is the top-left corner, W is width, and H is height.

OUTPUT FORMAT (exactly as shown):
[
  {{
    "id": "<unique_id>",
    "reasoning": <brief 1-line non-generic reasoning on choice of class>,
    "class": "<category_name>",
    "bbox": <Length 4 list [X, Y, W, H] where (X, Y) represents left-upper corner of bounding box and W and H width and height of the box respectively>,
    "confidence": <float 0-1>
  }}
]

COORDINATES:
- Origin (0,0) = top-left; units = pixels.
- Boxes tightly bound visible content, EXCEPT writable areas, which should include the full writable region.
- Overlap only when nested.
- Coordinates must be absolute pixel values expressed as floats (at most 4 decimal places).
- Do NOT use repeating decimal patterns.

FIELD RULES:
- **Headings**: page-level structural text such as headers, footers, titles, or section headers.
- **Fields**: any region related to user input, including both labels requesting information (keys) and the corresponding writable areas (values), as well as inline key-value blanks, checkboxes, tick marks, signature labels, signing areas, photo attachment placeholders, and other fillable elements.
- **Image**: purely graphical elements such as logos, icons, figures, or diagrams that are not intended for user input. Photo attachment placeholders are fields, not Image.
- **Table**: tabular structures with rows and/or columns, even if grid lines are faint or implicit.
- **Others**: instructional text, page numbers, scribbles, stray marks, or uncertain regions.

TABLE Specific RULES:
A. STRUCTURE
- Draw exactly one **Table** bounding box around the full table/grid region.
- Do NOT output internal rows, columns, headers, or cells.

B. RELATIONSHIPS
- Table captions or identifiers that are visually integrated with the table may be included inside the Table bounding box.

C. INVISIBLE GRIDS
- Infer alignment from consistent gutters or repeated leaders.
- Keep table edges straight and globally aligned.

OVERALL QUALITY RULES:
- Detect all visible fields; no omissions.
- Use visual layout, proximity, and language.
- Prefer meaningful, tightly bounded boxes.
- Do not output a single object unless the page truly contains only one visible element.
- Do not stop after the first detection if more than one visible element exists. Keep iterating until you have enumerated every visible field.
- Use “Others” for uncertain cases.
- Output must be in the described output format (a collection of JSON objects) — no other keys, natural text, or comments.

Full Image Size: Height -> {form_image_height}, Width -> {form_image_width}
Available Form Field Categories: {form_field_categories}
Use only the category names exactly as provided in 'Available Form Field Categories'. Do not invent, abbreviate, or modify category names.
Image: <image>
Your Layout Analysis Output:"""


DLA_FULL_COT_PROMPT = """You are a **form layout understanding AI**. Given an image of a form page, your task is to analyze its layout and identify all visible fields along with their categories and bounding boxes.
   input: 
   a. image of form page
   b. full image size (height and width)
   c. form field categories
   
   TASKS:
   1. Localize every visible field on the page.
   2. Classify each field into one of the given form field categories.
   3. Return bounding boxes [X, Y, W, H], where (X, Y) is the top-left corner, W is width, and H is height.
   
   OUTPUT FORMAT(exactly as shown):
   [
    {{
      "id": "<unique_id>",
      "reasoning": <brief 1-line non-generic reasoning on choice of class>,
      "class": "<category_name>",
      "bbox": <Length 4 list [X, Y, W, H] where (X, Y) represents left-upper corner of bounding box and W and H width and height of the box respectively>,
      "confidence": <float 0-1>
    }}
   ]

   COORDINATES:
   - Origin (0,0) = top-left; units = pixels.
   - Boxes tightly bound visible content, EXCEPT value fields, which should include the full writable area.
   - Overlap only when nested (e.g., key → value).
   - Coordinates must be absolute pixel values expressed as floats (at most 4 decimal places).
   - Do NOT use repeating decimal patterns.


   FIELD RULES:
   - **Header, Footer, Title of Form, Section Title, Page Num**: structural bands or labels.
   - **Form Key**: label requesting input (“Name”, “Date of Birth”).
   - **Form Value**: blank region for user entry; spacious bounding box.
   - **Inline Key, Inline Value**: key phrases and blanks within a sentence; value = writable area.
   - **Checkbox, Tick Mark**: selectable elements (“☐ Male”, “Yes / No”).
   - **Signature Key, Signature Val**: signature label and signing area (spacious box).
   - **Photo Field**: placeholders for attaching photos 
   - **Figure/Diagram/Logo**: icons. (This is one category covering figures, diagrams, and logos)
   - **Text Block**: instructions or narrative text.
   - **Gibberish, Mark for removal**: scribbles, stray marks.
   - **Others**: fallback for unclassified regions.

   TABLE RULES:
   A. STRUCTURE
   - Draw one **Table** box around the full grid region (even if lines are faint).
   - Choose one decomposition style:
    • **Column-major** → columns have headers at the top.  
    • **Row-major** → rows have labels on the left.  
    Do not mix both for one table.

   B. RELATIONSHIPS
   - Internal elements (keys, values, captions, section titles) set parent_id to the Table.
   - **Table Caption**: title above/below the table.
   - **Table Index**: short identifier referencing the table (e.g., “Table 1”, “Table II”, “Exhibit A”), usually placed immediately before the caption.
   - **Table Section Title**: header band dividing row/column groups.

   C. COLUMN-MAJOR
   - **Table Col PKey**: header cell areas across the top.
   - **Table Col Value**: vertical strips below headers spanning to table bottom.
   - Merged headers → one PKey linked to multiple Value regions
   - No overlap among adjacent columns.

   D. ROW-MAJOR
   - **Table Row PKey**: leftmost identifying cells for each row.
   - **Table Row Value**: remaining cells of that row.
   - Row Value spans full row height for multi-line content.

   E. MERGED / IRREGULAR CELLS
   - Assign each merged cell as key or value based on function.
   - Adjust boxes to include full merged area.

   F. INVISIBLE GRIDS
   - Infer alignment from consistent gutters or repeated leaders.
   - Keep column/row edges straight and globally aligned.

   G. CONSISTENCY
   - Keys = label regions; Values = data regions.
   - No overlaps among peers.
   - Preserve inner whitespace for writable cells.

   H. SECTION TITLES INSIDE TABLES
   - Bold or shaded header rows spanning multiple columns → Table Section Title.

   I. MULTI-PAGE TABLES
   - Annotate each page separately; repeat headers when they reappear.

   J. TEXT PREVIEW
   - Include short snippet if legible (“Name”, “Total”), else empty string.

   QUALITY RULES:
   - Detect all visible fields; no omissions.
   - Use visual layout, proximity, and language.
   - Prefer meaningful, tighly bounded boxes.
   - Do not output a single object unless the page truly contains only one visible element.
   - Do not stop after the first detection if more than one visible element exist. Keep iterating until you have enumerated every visible field.
   - Use “Others” for uncertain cases.
   - Output must be in described output format (a collection of JSON objects) — no other key, natural text or comments.
   
   Full Image Size: Height -> {form_image_height}, Width -> {form_image_width}
   Available Form Field Categories: {form_field_categories}
   Use only the category names exactly as provided in 'Available Form Field Categories'. Do not invent, abbreviate, or modify category names.
   Image: <image>
   Before producing the final output, internally reason step by step about the layout structure and label assignments, but do not include any intermediate reasoning in the output.
   Your Layout Analysis Output:"""


DLA_REDUCED_COT_PROMPT = """You are a **form layout understanding AI**. Given an image of a form page, your task is to analyze its layout and identify all visible fields along with their categories and bounding boxes.
input:
a. image of form page
b. full image size (height and width)
c. form field categories

TASKS:
1. Localize every visible field on the page.
2. Classify each field into one of the given form field categories.
3. Return bounding boxes [X, Y, W, H], where (X, Y) is the top-left corner, W is width, and H is height.

OUTPUT FORMAT (exactly as shown):
[
  {{
    "id": "<unique_id>",
    "reasoning": <brief 1-line non-generic reasoning on choice of class>,
    "class": "<category_name>",
    "bbox": <Length 4 list [X, Y, W, H] where (X, Y) represents left-upper corner of bounding box and W and H width and height of the box respectively>,
    "confidence": <float 0-1>
  }}
]

COORDINATES:
- Origin (0,0) = top-left; units = pixels.
- Boxes tightly bound visible content, EXCEPT writable areas, which should include the full writable region.
- Overlap only when nested.
- Coordinates must be absolute pixel values expressed as floats (at most 4 decimal places).
- Do NOT use repeating decimal patterns.

FIELD RULES:
- **Headings**: page-level structural text such as headers, footers, titles, or section headers.
- **Fields**: any region related to user input, including both labels requesting information (keys) and the corresponding writable areas (values), as well as inline key-value blanks, checkboxes, tick marks, signature labels, signing areas, photo attachment placeholders, and other fillable elements.
- **Image**: purely graphical elements such as logos, icons, figures, or diagrams that are not intended for user input. Photo attachment placeholders are fields, not Image.
- **Table**: tabular structures with rows and/or columns, even if grid lines are faint or implicit.
- **Others**: instructional text, page numbers, scribbles, stray marks, or uncertain regions.

TABLE Specific RULES:
A. STRUCTURE
- Draw exactly one **Table** bounding box around the full table/grid region.
- Do NOT output internal rows, columns, headers, or cells.

B. RELATIONSHIPS
- Table captions or identifiers that are visually integrated with the table may be included inside the Table bounding box.

C. INVISIBLE GRIDS
- Infer alignment from consistent gutters or repeated leaders.
- Keep table edges straight and globally aligned.

OVERALL QUALITY RULES:
- Detect all visible fields; no omissions.
- Use visual layout, proximity, and language.
- Prefer meaningful, tightly bounded boxes.
- Do not output a single object unless the page truly contains only one visible element.
- Do not stop after the first detection if more than one visible element exists. Keep iterating until you have enumerated every visible field.
- Use “Others” for uncertain cases.
- Output must be in the described output format (a collection of JSON objects) — no other keys, natural text, or comments.

Full Image Size: Height -> {form_image_height}, Width -> {form_image_width}
Available Form Field Categories: {form_field_categories}
Use only the category names exactly as provided in 'Available Form Field Categories'. Do not invent, abbreviate, or modify category names.
Image: <image>
Before producing the final output, internally reason step by step about the layout structure and label assignments, but do not include any intermediate reasoning in the output.
Your Layout Analysis Output:"""


KIE_ZS_PROMPT = """You are an information extraction system.
  Given:
  1. An image of a document.
  2. A question in the format: What is the value of <key_text>?

  Your task:
  - Locate the field corresponding to <key_text>.
  - Extract the exact visible value from the document.
  - If multiple possible values are found, return the one closest to the key text in the document layout.

  Output rules:
  - Return the answer strictly inside these tags:
    <answer>the exact extracted value</answer>
  - If the key is not present:
    <answer>NOT_FOUND</answer>
  - If the value is blank:
    <answer>EMPTY</answer>
  - Do NOT output anything outside these tags.
  - Do NOT explain.
  - Do NOT add extra text.

  Important:
  - Extract exactly as written (preserve formatting, symbols, punctuation).
  - Do NOT translate the value.
  - Do NOT switch the language of the extracted text.
  - Return the value exactly in the original script and language as it appears in the document.
  - Do not infer missing information.
  - Do not calculate or transform values.

  Example 1:
  If the document contains:
  নাম: রহমান

  And the question is:
  What is the value of নাম?

  Then the output must be:
  <answer>রহমান</answer>

  Example 2:
  If the document contains:
  Name: Miah

  And the question is:
  What is the value of Name?

  Then the output must be:
  <answer>Miah</answer>
  
  Now answer the following question using the rules above.

  Question: {question}
  Your answer:"""


def build_prompt(task_type, prompt_variant, label_set_variant="full", page_height=None, page_width=None, question=None):
    prompt = None
    if task_type == "dla":
        if prompt_variant == "zero_shot":
            if label_set_variant == "full":
                prompt = DLA_FULL_ZS_PROMPT.format(
                    form_image_height=page_height,
                    form_image_width=page_width,
                    form_field_categories=str(LONG_FORM_LABEL_CATEGORIES),
                )
            elif label_set_variant == "reduced":
                prompt = DLA_REDUCED_ZS_PROMPT.format(
                    form_image_height=page_height,
                    form_image_width=page_width,
                    form_field_categories=str(SHORT_FORM_LABEL_CATEGORIES),
                )
        elif prompt_variant == "cot":
            if label_set_variant == "full":
                prompt = DLA_FULL_COT_PROMPT.format(
                    form_image_height=page_height,
                    form_image_width=page_width,
                    form_field_categories=str(LONG_FORM_LABEL_CATEGORIES),
                )
            elif label_set_variant == "reduced":
                prompt = DLA_REDUCED_COT_PROMPT.format(
                    form_image_height=page_height,
                    form_image_width=page_width,
                    form_field_categories=str(SHORT_FORM_LABEL_CATEGORIES),
                )
    elif task_type == "kie":
        if prompt_variant == "zero_shot":
            prompt = KIE_ZS_PROMPT.format(question=question)
    return prompt