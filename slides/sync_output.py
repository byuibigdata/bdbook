# Slides post-render hook: copy the built site (slides/_site) into docs/slides.
# Building inside the slides project lets Quarto share one site_libs folder
# across all decks; building straight into ../docs/slides gives every deck
# its own <deck>_files/libs copy instead.
import os
import shutil

src = "_site"
dest = os.path.join("..", "docs", "slides")

if os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1":
    # Full render: _site is complete, so mirror it exactly.
    shutil.rmtree(dest, ignore_errors=True)
shutil.copytree(src, dest, dirs_exist_ok=True)
