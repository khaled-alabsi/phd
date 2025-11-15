from email.policy import default
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote
import zipfile
from datetime import datetime




def is_cjk_char(c):
    code_point = ord(c)
    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Extension A
        (0x20000, 0x2A6DF),  # CJK Extension B
        (0x2A700, 0x2B73F),  # CJK Extension C
        (0x2B740, 0x2B81F),  # CJK Extension D
        (0x2B820, 0x2CEAF),  # CJK Extension E
        (0x2CEB0, 0x2EBEF),  # CJK Extension F
        (0x2F800, 0x2FA1F),  # CJK Compatibility Supplement
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0xFF00, 0xFFEF),  # Full-width characters
    ]
    return any(start <= code_point <= end for start, end in cjk_ranges)


def _clean_cjk(text):
    cleaned = []
    for idx, char in enumerate(text):
        if is_cjk_char(char):
            print(f"Removed: Char '{char}' (U+{ord(char):04X}) at position {idx}", file=sys.stderr)
            continue
        cleaned.append(char)
    return ''.join(cleaned)


def _debug_line(file_name, step: int, line_num: int, old: str, new: str) -> None:
    if old != new:
        print(f"File: {file_name}")
        print(f"Step {step}, Line {line_num + 1}:")
        print("Before:")
        print(old.strip())
        print("After:")
        print(new.strip())
        print("-" * 50)


def fix_latex(path: str, show_changes: bool = False, overwrite_original_file: bool = False) -> None:
    print(f"Processing path: {path}")
    output_base = "./output"
    # Walk through all files in the directory and subdirectories
    for root, _, files in os.walk(path):
        print(f"Processing folder: {root}")
        for file_name in files:
            if file_name.endswith(".md"):
                input_file = os.path.join(root, file_name)
                print(f"Processing file: {input_file}")
                # Read the content of the markdown file
                with open(input_file, "r", encoding="utf-8") as file:
                    content = file.readlines()

                # Original content for comparison
                original_content = content[:]

                # Process each line individually
                for step, (pattern, replacement) in enumerate(
                    [
                        (r"\\\[", "$$"),
                        (r"\\\]", "$$"),
                        (r"\\\(", "$"),
                        (r"\\\)", "$"),
                        (r"(\$\$)(\*\*)", r"\1\n\2"),
                        (r"(\$\$)(###)", r"\1\n\2"),
                        (r"(\$\$)(\S)", r"\1\n\2"),
                        (
                            r"(?<!\S)\$(\s*[^\$]+?\s*)\$(?![\*\.$])",
                            r"$\1$",
                        ),  # Fixed Step 8
                        (r"(?<!\S)\$(?!\s|\$)", r"$"),  # Ensure single $
                        #(r"\$(?![\*\.$\s])", r"$ "),  # Ensure space after $
                        (r"\*\* \$(.*?)\$\*\*", r"**$\1$**"),
                    ],
                    start=1,
                ):
                    for i, line in enumerate(content):
                        new_line = re.sub(pattern, replacement, line)
                        if show_changes:
                            _debug_line(file_name, step, i, line, new_line)
                        content[i] = new_line

                # Inline block to sanitize LaTeX expressions
                latex_pattern = re.compile(r"(\$\s*)(.*?)(\s*\$)")
                for i, line in enumerate(content):
                    new_line = latex_pattern.sub(lambda m: f"${m.group(2).strip()}$", line)
                    if show_changes:
                        _debug_line(file_name, "Sanitize LaTeX", i, line, new_line)
                    content[i] = new_line
                # remove CJK characters
                for i, line in enumerate(content):
                    new_line = _clean_cjk(line)
                    if show_changes:
                        _debug_line(file_name, "Clean CJK", i, line, new_line)
                    content[i] = new_line
                if content != original_content:
                    if overwrite_original_file:
                        # Overwrite the original file
                        with open(input_file, "w", encoding="utf-8") as file:
                            file.writelines(content)
                        print(f"File overwritten: {input_file}")
                    else:
                        # Create output folder structure: output_base/test/<relative_path>/file_name
                        relative_path = os.path.relpath(input_file, path)
                        output_file = os.path.join(output_base, relative_path)
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        # Write the updated content to the output folder
                        with open(output_file, "w", encoding="utf-8") as file:
                            file.writelines(content)
                        print(f"File written to: {output_file}")


def _update_image_paths(md_content: str, md_file_path: Path) -> str:
    """
    Update image paths in a Markdown file to full paths based on its current location.

    Args:
        md_content (str): The content of the Markdown file.
        md_file_path (Path): The path of the Markdown file.

    Returns:
        str: The updated Markdown content with corrected image paths.
    """
    image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")

    matches = image_pattern.findall(md_content)
    if matches:
        for match in matches:
            original_path = match
            full_path = (md_file_path.parent / original_path).resolve()
            md_content = md_content.replace(original_path, str(full_path))
    return md_content


def _sanitize_latex_expressions(md_content: str) -> str:
    """
    Sanitize LaTeX expressions by removing spaces after the first $
    and before the last $. Also, ensure that '---' is always surrounded
    by two new lines.

    Args:
        md_content (str): The content of the Markdown file.

    Returns:
        str: The updated Markdown content with sanitized LaTeX expressions
             and properly formatted '---' separators.
    """
    # Sanitize LaTeX expressions
    latex_pattern = re.compile(r"(\$\s*)(.*?)(\s*\$)")
    md_content = latex_pattern.sub(lambda m: f"${m.group(2).strip()}$",
                                   md_content)

    # Ensure '---' is always surrounded by exactly two new lines
    md_content = re.sub(r"\n*---\n*", "\n\n---\n\n", md_content)

    return md_content


def _update_first_line(md_content: str) -> str:
    """
    Update the first line of the Markdown content if it starts with ###.

    Args:
        md_content (str): The content of the Markdown file.

    Returns:
        str: The updated Markdown content with the first line modified if needed.
    """
    lines = md_content.splitlines()
    if lines and lines[0].startswith("###"):
        lines[0] = lines[0].replace("###", "##", 1)
    return "\n".join(lines)


def combine_md_files(input_folder: str, output_file: str,
                     output_folder: str) -> None:
    combined_content = []
    input_folder_path = Path(input_folder)
    output_folder_path = Path(output_folder)

    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True)

    # Collect all Markdown files and sort them by their names
    md_files = []
    for root, _, files in os.walk(input_folder_path):
        for file in files:
            if file.endswith(".md"):
                md_files.append(Path(root) / file)

    # Sort the files by name
    md_files.sort(key=lambda x: x.name)

    print("\nFiles will be added in this order:")
    for idx, file_path in enumerate(md_files, start=1):
        print(f"{idx}. {file_path.name}")

    # Process each file
    for file_path in md_files:
        print(f"\nAdding file to combined content: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Update the first line if it starts with ###
        content = _update_first_line(content)
        # Update image paths
        content = _update_image_paths(content, file_path)
        # Sanitize LaTeX expressions
        content = _sanitize_latex_expressions(content)
        combined_content.append(content)

    # Write the combined content to the output file
    output_path = output_folder_path / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_content))

    print(f"\nCombined content written to: {output_path}")


def combine_with_subfolders(input_folder: str,
                            final_combined_output_path: str) -> None:
    """
    Process all subfolders in the input folder, combining Markdown files from each subfolder
    into a single Markdown file named after the subfolder. Then, combine these subfolder-level
    files into a final combined Markdown file.

    Args:
        input_folder (str): The root input folder containing subfolders.
        final_combined_output_path (str): The path where the final combined Markdown file will be saved.
    """
    input_folder_path = Path(input_folder)

    # Iterate over subfolders in the input folder
    for subfolder in input_folder_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            subfolder_output_file = input_folder_path / f"{subfolder_name}.md"

            # Check if subfolder contains Markdown files
            md_files = list(subfolder.rglob("*.md"))
            if md_files:
                print(f"\nProcessing subfolder: {subfolder}")
                combine_md_files(str(subfolder), subfolder_output_file.name,
                                 input_folder)

    # After processing subfolders, combine all subfolder-level combined files into one final file
    print(
        "\nCombining all subfolder-level combined files into one final file..."
    )
    combine_md_files(
        input_folder,
        Path(final_combined_output_path).name,
        str(Path(final_combined_output_path).parent),
    )


def generate_content_files(root_path: str,
                           use_root_relative: bool = False) -> None:
    root = Path(root_path)

    for folder in root.rglob('*'):
        if folder.is_dir():
            md_files = [
                f for f in folder.glob('*.md') if f.name != 'content.md'
            ]
            if md_files:
                content_file = folder / 'content.md'
                header = f"### {folder.name}\n\n"
                lines = []
                for f in md_files:
                    rel_base = root if use_root_relative else folder
                    rel_path = f.relative_to(rel_base)
                    link = quote(str(rel_path).replace("\\", "/"))
                    lines.append(f"- [{f.stem}]({link})")
                content = header + "\n".join(lines) + "\n"
                content_file.write_text(content, encoding='utf-8')


def aggregate_content_files(root_path: str,
                            output_filename: str = 'content.md') -> None:
    root = Path(root_path)
    all_content_files = sorted(root.rglob(output_filename))
    output_file = root / output_filename

    lines = []
    for content_file in all_content_files:
        if content_file.resolve() == output_file.resolve():
            continue
        #header = f"# {content_file.parent.name} ({content_file.parent.relative_to(root)})\n\n"
        text = content_file.read_text(encoding='utf-8')
        lines.append(text.strip() + "\n")

    output_file.write_text("\n".join(lines), encoding='utf-8')



defualt_excluded_dirs= ["./archive","./.git","./.vscode", "./.venv"]
defualt_excluded_dir_names: list[str] = [".vscode","__pycache__","images","assets"] 
defualt_excluded_file_extensions: list[str] = []
defualt_excluded_files_names: list[str] = [".git",".venv",".vscode","images","assets",".gitignore","content.md", ".DS_Store","requirements.txt","tools.ipynb","update table of content.ipynb"]

def create_main_content_file(
        root_input_path: str = ".",
        output_file_name: str = "Table of content.md",
        output_file_dir: str = ".",
        excluded_dirs: list[str] = defualt_excluded_dirs,
        excluded_dir_names: list[str] = defualt_excluded_dir_names, 
        excluded_file_extensions: list[str] = [],
        excluded_files_names: list[str] = defualt_excluded_files_names) -> None:
    # Resolve root and output paths
    root = Path(root_input_path).resolve()
    output_path = Path(output_file_dir).resolve() / output_file_name

    # Normalize excluded directories to absolute paths
    excluded_dirs_resolved = [Path(excl).resolve() for excl in excluded_dirs]

    lines: list[str] = []
    lines.append("# Tables of content\n\n")

    def add_to_lines(path: Path, level: int) -> None:
        indent = "  " * level
        relative_path = quote(str(path.relative_to(root)).replace("\\", "/"))
        lines.append(f"{indent}- [{path.stem}]({relative_path})")

    def process_directory(directory: Path, level: int) -> None:
        # Separate files and subdirectories in the current directory
        files = sorted([
            file for file in directory.iterdir()
            if file.is_file() #and file.suffix == ".md"
        ])
        subdirs = sorted(
            [subdir for subdir in directory.iterdir() if subdir.is_dir()])

        # Add files first
        for file in files:
            # Skip excluded files
            if file.name in excluded_files_names or file.suffix in excluded_file_extensions:
                continue
            add_to_lines(file, level)

        # Recursively process subdirectories
        for subdir in subdirs:
            # Skip excluded directories by name
            if subdir.name in excluded_dir_names:
                continue

            # Skip excluded directories by path
            if any(subdir.resolve().is_relative_to(excl)
                   for excl in excluded_dirs_resolved):
                continue

            # Add directory header
            lines.append(f"{'  ' * level}- {subdir.name}")
            # Process the subdirectory recursively
            process_directory(subdir, level + 1)

    # Start processing from the root directory
    process_directory(root, level=0)

    # Write the output file
    output_path.write_text("\n".join(lines), encoding="utf-8")

def create_directory_structure_md(
    root_input_path: str = ".",
    output_file_name: str = "Project Structure.md",
    output_file_dir: str = ".",
    excluded_dirs: list[str] = defualt_excluded_dirs,
    excluded_file_extensions: list[str] = [],
    ignore_folder_names: list[str] = defualt_excluded_files_names
) -> None:
    root = Path(root_input_path).resolve()
    output_path = Path(output_file_dir).resolve() / output_file_name

    lines: list[str] = []

    def walk_dir(path: Path, level: int) -> None:
        indent = "│   " * level
        for item in sorted(path.iterdir()):
            if any(ign in item.name for ign in ignore_folder_names):
                continue
            if item.name in excluded_dirs or any(part in excluded_dirs for part in item.parts):
                continue
            if item.is_dir():
                lines.append(f"{indent}├── {item.name}/")
                walk_dir(item, level + 1)
            elif item.is_file():
                if item.suffix in excluded_file_extensions:
                    continue
                rel_path = quote(
                    str(item.relative_to(root)).replace("\\", "/"))
                lines.append(f"{indent}├── [{item.name}]({rel_path})")

    lines.append(f"{Path(root).name}/")
    walk_dir(root, 0)

    output_path.write_text("\n".join(lines), encoding="utf-8")

def backup_as_zip(input_path='.', output_path='./archive'):
    """
    Zips the contents of the given folder (default is the current directory)
    into a SINGLE zip file and saves it in the specified output directory 
    with a timestamped name for version control.
    
    Parameters:
        input_path (str): Path to the folder to be zipped. Default is the current directory ('.').
        output_path (str): Path to the directory where the zip file will be saved. Default is './archive'.
    """
    # Ensure the input path is absolute
    input_path = os.path.abspath(input_path)

    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)

    # Generate a timestamped name for the zip file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    zip_filename = f"backup_{timestamp}.zip"
    zip_filepath = os.path.join(output_path, zip_filename)

    # Get the absolute path of the output directory to exclude it
    output_dir_abs = os.path.abspath(output_path)

    # Create a SINGLE zip file
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the input directory
        for root, dirs, files in os.walk(input_path):
            # Exclude the output directory from being zipped
            if os.path.commonpath([root, output_dir_abs]) == output_dir_abs:
                continue

            for file in files:
                file_path = os.path.join(root, file)

                # Add the file to the zip archive with its relative path
                arcname = os.path.relpath(file_path, input_path)
                zipf.write(file_path, arcname)

    print(f"Backup created successfully: {zip_filepath}")
