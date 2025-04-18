# GitHub Copilot Custom Instructions

## What would you like Copilot to know about you to provide better responses?

I am working on an early-stage image classification project using convolutional neural networks (CNNs). My goal is to determine whether my dataset of a few thousand images is usable before investing in a more complex pipeline.

To achieve this, I am using:
- **FastAI** for training CNN models quickly with minimal code.
- **OpenCV** for basic image visualization.
- **Label Studio** for dataset annotation and exploration.
- **Python** as my primary development environment.

I prioritize:
- **Readable, modular code** with comments explaining key steps.
- **Simplicity over performance** (for now), focusing on quick validation of my data.
- **Pre-trained models (like ResNet)** for transfer learning instead of training from scratch.
- **Visual outputs** where possible (e.g., `.show_batch()`, confusion matrices, and OpenCV plots).

Copilot should:
- Suggest **FastAI-friendly** code when working with image classification.
- Prefer **Python-friendly** code (e.g., `matplotlib.pyplot` for plots, `print()` instead of logging).
- Use **OpenCV (`cv2`)** for image display instead of unnecessary libraries.
- Favor **pre-trained CNNs** like `resnet101` unless I explicitly request custom architectures.
- When suggesting dataset preprocessing, use `ImageDataLoaders.from_folder()` and **avoid overly complex manual pipelines**.
- When suggesting image augmentation, use **FastAI transforms**, not raw NumPy operations.
- Add dependencies to requirements.txt when you need them

I would also like Copilot to provide short, **inline comments** explaining key logic.

## How would you like Copilot to respond?

- Respond with **structured, step-by-step code snippets** instead of single-line solutions.
- Provide **explanations** in code comments, but keep responses concise.
- Suggest **function-based** implementations instead of monolithic scripts.
- Offer **Python-friendly** visual outputs when applicable (e.g., `.show_batch()`, OpenCV, or `matplotlib` plots).
- If recommending additional libraries, ensure they are **lightweight and relevant** to my workflow.
- If multiple approaches exist, briefly mention the alternatives but default to **FastAI’s high-level API**.

When troubleshooting errors:
- Prioritize **interpretable debugging steps** (e.g., checking `dls.valid.show_batch()` for bad labels).
- Suggest **practical fixes** before proposing deep-dive solutions.
- Avoid over-complicating things with unnecessary optimizations.

If I need further refinements, Copilot should ask **clarifying questions** instead of making assumptions.


## Maintenance Instructions

### Keeping Documentation in Sync

When making significant changes to the codebase:
- **Adding new functions**: Update the corresponding module section in this file
- **Renaming functions**: Update any references to the old function name in this file
- **Changing function parameters**: Update any examples or descriptions that reference the parameters
- **Removing functions**: Remove mentions from this document to avoid confusion
- **Adding new modules**: Create a new section in the "Project File Structure" area with relevant details

This ensures that Copilot has accurate information about the codebase structure when providing assistance.

### Working with Existing Files

When making changes to the codebase:
- If you need access to a file that isn't visible, ask for it explicitly
- Don't create placeholder implementations for files that should already exist
- If suggesting new files, explicitly indicate where they should be placed in the existing structure
- Always specify relative paths when referring to existing files (e.g., ./utils/model_utils.py)
