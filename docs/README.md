How to generate docs using sphinx.

Ensure your code is documented using python docstrings.

create a env using for python Python 3.12.3 
'''
python3.12.3 -m venv {Your name for the env}
'''

Then activate the env
'''
source /{Your name for the env}/bin/activate
'''
Then ensure you are inside of the docs directory. It should include /build /source and files make.bat, Makefile, build_docs.sh

Then once you are in the docs directory with the env active you can run 
'''
pip install -r requirements.txt
'''
This will ensure you have the nessecary dependencies.

Now that you have done the prep you can generate the documenation using either of these two commands.
'''
make latexpdf
'''
or
'''
make html
'''
These will generate either a pdf or an html