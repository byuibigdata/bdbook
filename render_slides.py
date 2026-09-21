# Quarto post-render hook for the book.
# A full book render cleans docs/, which deletes docs/slides. Re-render the
# slides project afterward so the class slides stay published. Single-file
# renders and previews don't clean docs/, so skip the extra work then.
import os
import subprocess

if os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1":
    subprocess.run(["quarto", "render", "slides"], check=True)
