# #!/bin/bash
# # Navigate to docs folder
# cd "$(dirname "$0")"

# echo "Cleaning old build..."
# make clean

# echo "Generating RST files from source..."
# sphinx-apidoc -f -o source/modules ../src

# echo "Building HTML docs..."
# make html

# echo "Done! Open docs/build/html/index.html to view."
#!/bin/bash
# Navigate to docs folder
# cd "$(dirname "$0")"

# echo "Cleaning old build..."
# make clean

# echo "Generating RST files from source (submodules only)..."

# # Generate .rst files for all submodules of src, but exclude the top-level src package itself
# sphinx-apidoc -f -o source/modules ../src \
#     --separate \
#     --module-first \
#     --no-toc

# echo "Building HTML docs..."
# make html

# echo "Done! Open docs/build/html/index.html to view."

#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")/docs"

echo "Cleaning old build..."
make clean

# Path to the src.rst that we want to preserve
TARGET_SRC="source/modules/src.rst"

# Temporarily back up src.rst if it exists
if [ -f "$TARGET_SRC" ]; then
    TMP_BACKUP="$(mktemp)"
    cp "$TARGET_SRC" "$TMP_BACKUP"
    echo "Backed up src.rst"
fi

echo "Generating RST files from source (submodules only)..."

# Generate .rst files for all submodules of src, excluding the top-level src package itself
# sphinx-apidoc -f -o source/modules ../src \
#     --separate \
#     --module-first \
#     --no-toc
sphinx-apidoc -f -o source/modules ../src \
    --separate \
    --module-first \
    --no-toc \
    # --exclude ../src/__init__.py

# Restore src.rst if it was backed up
if [ -f "$TMP_BACKUP" ]; then
    cp "$TMP_BACKUP" "$TARGET_SRC"
    rm "$TMP_BACKUP"
    echo "Restored src.rst"
fi

echo "Building HTML docs..."
make html

echo "Done! Open docs/build/html/index.html to view."