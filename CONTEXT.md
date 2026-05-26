# AI Tools

Small command-line utilities for converting, extracting, counting, and processing local media and document files.

## Language

**Narration Audio**:
An audio file containing spoken narration generated from readable document content. Markdown syntax is not read aloud except when the user explicitly asks for literal reading.
_Avoid_: raw Markdown audio, literal Markdown audio

**Markdown Document**:
A Markdown file used as the source document for conversion into another medium.
_Avoid_: markdown file, md input

**Markdown Narration Tool**:
The command-line tool that converts a Markdown Document into a Narration File.
_Avoid_: markdown-to-speech app, narrator

**Non-Prose Block**:
A part of a Markdown Document that is not naturally spoken prose, such as a code block, table, embedded HTML, or image. In Narration Audio, it is represented by a short omission cue rather than being read verbatim.
_Avoid_: technical block, markdown artifact

**Rendered Prose**:
The speakable text extracted from a Markdown Document after formatting syntax is removed. Headings and list items remain as text, links are represented by visible link text only, footnotes are omitted, and structure is conveyed through pauses rather than labels such as "section" or "bullet".
_Avoid_: raw Markdown, flattened text

**Narration Preview**:
A text-only view of the Rendered Prose chunks that would be spoken, used before creating a Narration File.
_Avoid_: dry output, debug text

**Narration Language**:
The spoken language used for Narration Audio. This tool supports English and Brazilian Portuguese only, and the user must choose one explicitly.
_Avoid_: arbitrary language, auto-detected language

**Narration Voice**:
The built-in speaker voice used to produce Narration Audio. A Narration Voice does not require the user to provide a reference recording, and the Markdown Narration Tool provides a default while allowing the user to choose another supported voice.
_Avoid_: cloned voice, reference voice

**Narration Instruction**:
Optional user guidance for the speaking style of Narration Audio, such as tone, pace, or presentation style.
_Avoid_: prompt, system prompt

**Combined Narration**:
A single Narration Audio file representing the whole Markdown Document, even if the document is processed internally in smaller parts.
_Avoid_: chapter files, section exports

**Narration File**:
The generated audio file containing Combined Narration. The default Narration File format is WAV.
_Avoid_: export, render

## Example Dialogue

Dev: "Should the Markdown symbols be spoken?"

Domain expert: "No. The Markdown Document is the source, but the output is Narration Audio, so headings, lists, and paragraphs should sound natural."

Dev: "What do we call the CLI?"

Domain expert: "It is the Markdown Narration Tool. In the repository it follows the existing short converter naming style."

Dev: "What about code blocks and tables?"

Domain expert: "Those are Non-Prose Blocks. The narration should mention that they were omitted rather than trying to read them aloud."

Dev: "Should list bullets be spoken?"

Domain expert: "No. Rendered Prose keeps the item text and uses pauses to make the structure audible."

Dev: "Can the user inspect what will be spoken before generating audio?"

Domain expert: "Yes. Narration Preview shows the Rendered Prose chunks without creating a Narration File."

Dev: "Can the tool narrate any language?"

Domain expert: "No. Narration Language is limited to English and Brazilian Portuguese, and the user must choose it explicitly."

Dev: "Does the user need to provide a sample recording?"

Domain expert: "No. The first version uses a Narration Voice, so the Markdown Document and selected language are enough."

Dev: "Can the user ask for a calmer or lecture-like delivery?"

Domain expert: "Yes. They can provide a Narration Instruction, but neutral narration is the default."

Dev: "Should each heading become its own audio file?"

Domain expert: "No. The expected output is Combined Narration: one audio file for the whole Markdown Document."

Dev: "Should the output be MP3?"

Domain expert: "The default Narration File is WAV because it is the most reliable synthesis target."
