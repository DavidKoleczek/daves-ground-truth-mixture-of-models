INITIAL_IMAGE_TEMPLATE = """\
{{ infographic_prompt }}

Please place the infographic in {{ image_abspath }}"""

EVALUATION_PROMPT_TEMPLATE = """\
I want you to evaluate and decide if you should refine an infographic that was generated from the following prompt:

{{ original_prompt }}

The infographic image is at `{{ image_path }}`. Look at it carefully.

First, evaluate it through the following lens.

{{ criterion_prompt }}

Then decide how to prompt Gemini's Nano Banana image editing tool to improve the infographic.
You can take big swings at editing, the image editor is powerful! The initial image might not be that great.
This can include asking for a few things at once, or asking it to re-create the image overall with a new prompt.

To edit, run the following shell command, replacing the placeholder with your specific edit instructions based on your evaluation:
```
gemini -p 'Use the edit_image tool to edit the file {{ image_abspath }}. \
<your specific edit instructions here>' \
-m gemini-3.1-pro-preview -o json --approval-mode yolo
```

After the edit completes, copy the newest file from `nanobanana-output/` back to `{{ image_abspath }}` so the user sees the updated image. 
Override the original image with the new one, do not create a new file.

In some cases, it might not be necessary to edit the image.
In that case briefly state why the infographic already meets everything that the original prompt requested.

Only edit the image once, and once you don you don't have to look at it again."""

INFORMATION_ARCHITECTURE_CRITERION = """\
Evaluate from the perspective of INFORMATION ARCHITECTURE AND CONTENT ACCURACY.

Focus on these questions:
- Is all the content requested in the original prompt present and accurately represented? \
Look for missing sections, garbled or duplicated text, nonsensical labels, or factual errors.
- Is there a clear visual hierarchy that guides the viewer's eye through the information \
(e.g., headline > subheading > body > caption)? Does the layout make the most important \
information the most prominent, or could the hierarchy be stronger?
- Does the layout follow a logical reading flow that matches the intended narrative order? \
Can a viewer intuitively follow the sequence without getting lost?
- Are sections and concepts clearly delineated so the viewer knows where one idea ends \
and the next begins? Are there sections that could be split, merged, or reordered to tell \
the story more clearly?
- Could labels, headings, or captions be rewritten to be more concise and scannable?
- Apply the 5-second test: could someone identify the main takeaway within 5 seconds of \
looking at this? If not, what needs to change to make the core message immediately obvious?
- Is there unnecessary complexity or information that does not serve the core message? \
Everything included should earn its place -- cut anything that is noise rather than signal.

Do NOT comment on color choices, icon style, or overall aesthetics as those are outside your scope."""


VISUAL_COMMUNICATION_CRITERION = """\
Evaluate from the perspective of VISUAL COMMUNICATION EFFECTIVENESS.

Focus on these questions:
- Do visual elements (icons, diagrams, illustrations, visual metaphors) actively explain \
the concepts, or are they purely decorative?
- Are there text-heavy sections that could be replaced or supplemented with visual \
representations? For example, could a process described in text become a flowchart, \
a comparison become a side-by-side diagram, or a list become a visual timeline?
- Is there a visual metaphor or analogy that would make a concept click faster for the \
target audience?
- Would adding visual connections between sections (arrows, numbered paths, color-coded \
groupings) improve the narrative flow and make the sequence of information clearer?
- Does the infographic tell a visual story with a narrative arc (context, evidence, insight), \
or does it read like a poster of disconnected facts? A great infographic is a visual \
argument, not just a collection of information.
- Does the viewer have to work hard to parse the information, or does understanding come \
naturally? If the cognitive load is high, what could be simplified or restructured visually?
- Are key insights visually distinctive enough to be memorable after viewing? Would the \
target audience recall the main points an hour later?

Do NOT comment on color harmony, font choices, or whether content is factually complete as those are outside your scope."""

VISUAL_DESIGN_CRITERION = """\
Evaluate from the perspective of VISUAL DESIGN AND AESTHETIC COHESION.

Focus on these questions:
- Is the color palette cohesive and purposeful? Are colors used consistently to encode meaning \
rather than applied arbitrarily?
- Is typography readable with a clear hierarchy? Are font sizes, weights, and styles used \
to distinguish headlines from body text from labels?
- Is whitespace used effectively to prevent visual clutter and give elements room to breathe?
- Does the overall style feel consistent throughout? Check icon rendering, border treatment, \
background texture, and any illustrative elements for stylistic coherence.
- Is there sufficient contrast between text and background for all elements to be legible?
- Are there any common AI generation artifacts? Look for: overlapping elements, misaligned \
sections, text that bleeds outside its bounding box, elements that clip the image edges, \
inconsistent spacing between similar items, or objects that float in unnatural positions.

Do NOT comment on whether the content is complete or the narrative flow is logical as those are outside your scope."""

CRITERIA: dict[str, str] = {
    "information_architecture": INFORMATION_ARCHITECTURE_CRITERION,
    "visual_design": VISUAL_DESIGN_CRITERION,
    "visual_communication": VISUAL_COMMUNICATION_CRITERION,
}
