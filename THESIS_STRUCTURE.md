# THESIS_STRUCTURE.md — thesis-by-publication + LaTeX repo spec (Overleaf + Quantum article)

Areas of Interest: Compiler Design, Programming Languages, Quantum Programming
Date Added: March 20, 2026 6:54 PM (GMT+11)
Description: Repo/LaTeX structure spec for thesis-by-publication (paper-as-chapter), with naming conventions and conversion plan; intended to guide an agent setting up a GitHub+Overleaf-compatible repo.
Type: Notes

## Purpose

A markdown specification for structuring a PhD thesis-by-publication and a composable LaTeX + GitHub repository.

## THESIS_[STRUCTURE.md](http://STRUCTURE.md)

# Thesis-by-Publication Structure + Composable LaTeX Repo Spec (Paper-as-Chapter, Overleaf-ready)

## Goal

Create a PhD thesis-by-publication in LaTeX where:

- Each thesis chapter is split into a folder and multiple `.tex` files (one per logical section/subsection) for composability.
- Included-publication chapters (the “paper chapters”) are “paper-as-chapter”: the chapter is primarily the included publication, wrapped by minimal thesis-specific material to avoid duplication.
- Naming does not encode numbering (LaTeX numbering is dynamic).
- The repository is compatible with Overleaf (single root `main.tex`, no nonstandard build system required).
- Each publication can also be compiled as a standalone paper using a quantum journal article style (typically `quantumarticle` from the Quantum journal).

## Naming conventions (strict)

No numeric identifiers in directory or file names.

- Chapters: `thesis/chapters/chapt_{title_slug}/`
- Included-publication chapters: `thesis/chapters/chapt_pub_{unique_term}/`
- Chapter section files: `sec_{title_slug}.tex`
- Publications: `publications/publication_{unique_term}/`

Where:

- `{title_slug}` is a short lowercase slug using letters/numbers/underscores only.
- `{unique_term}` is a stable, distinctive identifier for a publication (never `1`, `2`, etc.).

Examples:

- `thesis/chapters/chapt_introduction/sec_motivation.tex`
- `thesis/chapters/chapt_pub_operations_model/preface.tex`
- `publications/publication_operations_model/sec_related_work.tex`

## Repository layout (agent-readable, Overleaf-compatible)

Top-level:

- `README.md`
- `THESIS_STRUCTURE.md`
- `preamble/` (Shared LaTeX configuration)
    - `packages.tex` (Global package imports, e.g., `import`, `booktabs`)
    - `macros.tex`
    - `bibliography.tex`
- `thesis/` (The main thesis project)
    - `main.tex` (thesis build entrypoint; Overleaf compiles this)
    - `frontmatter/`
        - `titlepage.tex`
        - `abstract.tex`
        - `declaration.tex`
        - `acknowledgements.tex`
        - `preface_publications.tex` (list of included publications + authorship notes)
    - `chapters/`
        - `chapt_introduction/`
        - `chapt_literature_review/`
        - `chapt_methodology/`
        - `chapt_pub_operations_model/`
        - `chapt_pub_language_design/`
        - `chapt_pub_compilation_dynamic_static/`
        - `chapt_results_evaluation/`
        - `chapt_study_implications/`
        - `chapt_future_directions_conclusion/`
- `publications/` (Standalone publication sources)
    - `publication_operations_model/`
        - `main.tex` (the standalone journal build entrypoint)
        - `content.tex` (the shared paper content)
        - `sec_{section_slug}.tex` (individual sections)
    - `publication_language_design/`
    - `publication_compilation_dynamic_static/`
- `figures/`
- `tables/`
- `bib/references.bib`

## Paper-as-chapter rule (for the included-publication chapters)

For the included-publication chapters:

- Do NOT re-explain the paper in long thesis prose sections.
- Keep the chapter wrapper limited to:
    - a brief chapter overview (thesis-level positioning),
    - publication metadata + contribution statement + linking commentary,
    - the included publication content itself (via `content.tex`),
    - a short transition paragraph (optional).

All detailed related work, definitions, methods, and results should live in the paper content, not duplicated outside it.

## Publication representation (dual-use)

Each included publication lives at:

- `publications/publication_{unique_term}/` (the flattened “paper text”)
    - `main.tex` (the standalone journal build entrypoint, includes `content.tex`)
    - `content.tex` (the paper content entrypoint, includes the sections)
    - `sec_{section_slug}.tex` (individual sections)

Thesis-specific wrappers for the publication live at:

- `thesis/chapters/chapt_pub_{unique_term}/` (shared with the chapter overview)
    - `preface.tex` (citation, venue, publication status)
    - `contribution_statement.tex` (authorship / your contribution)
    - `linking_commentary.tex` (how it fits the thesis; what differs vs published version)
    - `include.tex` (stable include interface for the thesis chapter; uses `\subimport` to resolve relative paths in the paper content)

Additionally, ensure advisory section files exist under `publications/publication_{unique_term}/` even before they’re finalized (they can be empty placeholders initially), e.g. `sec_introduction.tex`, `sec_related_work.tex`, etc.

`thesis/chapters/chapt_pub_{unique_term}/include.tex` must be the stable interface that the thesis chapter inputs. It should input the local wrapper files, then use `\subimport` to include the paper’s `publications/publication_{unique_term}/content.tex`. Using `\subimport` (from the `import` package) ensures that `\input` commands inside the paper's `content.tex` correctly resolve relative to the paper's directory, even when compiled as part of the main thesis.

## Quantum article style (Quantum journal) + Overleaf compatibility

Agent instructions:

- For each `publications/publication_{unique_term}/`, ensure `main.tex` uses the Quantum journal article style (`quantumarticle`).
- Download/include the Quantum journal LaTeX template/style files in-repo (so Overleaf builds without external downloads). Place them under:
    - `latex_styles/quantum/` (or similar)
- Ensure the thesis build (`/main.tex`) and paper builds do not conflict:
    - keep paper-specific class/style files in `latex_styles/` and reference via relative paths.
    - keep macros shared in `preamble/macros.tex` and input them from both thesis and paper builds as appropriate.

## Chapter folder template

Each chapter folder:

- `chapter.tex` (contains `\chapter{...}` and inputs its `sec_*.tex` files)
- multiple `sec_{title_slug}.tex` files (no numbering in filenames)

`chapter.tex` should explicitly list inputs in the intended reading order.

## Main include order (high level)

`main.tex` should:

- input preamble
- input frontmatter (roman numbering)
- input chapters (arabic numbering)
- bibliography
- appendices (optional)

---

# Thesis-level literature review chapter (Option 1)

Add a dedicated narrative chapter:

Folder: `thesis/chapters/chapt_literature_review/`

Suggested section files:

- `sec_overview.tex`
- `sec_quantum_computing_foundations.tex`
- `sec_quantum_programming_languages.tex`
- `sec_hybrid_quantum_classical_models.tex`
- `sec_quantum_compilation_and_irs.tex`
- `sec_accessibility_portability_usability.tex`
- `sec_synthesis_gap_and_requirements.tex`

---

# Conversion instructions (for later agent enactment)

This repo is designed so an agent can later convert the included-publication chapters from “paper-as-chapter” into “paper + extended thesis treatment” without imposing numeric filenames.

## Conversion goal

Add thesis-only material (extra background, expanded proofs, extra experiments, extended discussion) around the included paper content, while keeping:

- `publications/publication_{unique_term}/` as the stable “paper text”
- `thesis/chapters/chapt_pub_{unique_term}/include.tex` as the stable include interface

## Agent-enactable mechanical steps (per included-publication chapter)

For each included-publication chapter folder (named `chapt_pub_{unique_term}`):

1) Add new thesis-only section files (no numbering; name by title):

- `sec_extra_background_or_notation.tex` (optional)
- `sec_extended_methods_or_proofs.tex` (optional)
- `sec_extended_results_or_ablations.tex` (optional)
- `sec_postpaper_discussion.tex` (optional)

2) Update the chapter’s `chapter.tex` to input sections in a revised order:

- overview
- (optional) extra background / notation
- publication wrapper
- included publication (still via `thesis/chapters/chapt_pub_{unique_term}/include`)
- (optional) extended proofs/results/discussion
- transition

3) Avoid duplication:

- Keep `linking_commentary.tex` structural and brief.
- Put new detail only into the new thesis-only `sec_*.tex` files, not into the paper content.

## Optional formatting toggles (only if needed later)

If you must compile the same `paper_content` both standalone and inside the thesis with different formatting, an agent may introduce a boolean (e.g., `\newif\ifthesis`) in `preamble/macros.tex` and conditionally include formatting differences. Avoid this unless required.