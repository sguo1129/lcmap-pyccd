## Developing PyCCD

#### app.py

#### cli.py and entry_point scripts
The command line interface is implemented using the click project, which
provides decorators for functions that become command line arguments.

Integration with setup.py entry_point is done via click-plugins, which allow
cli commands to also be designated as entry point scripts.

See ccd.cli.py, setup.py and the click/click-plugin documentation.

[Click Docs](http://click.pocoo.org/5/)
[Click On Github](https://github.com/pallets/click)
[Click on PyPi](https://pypi.python.org/pypi/click)
[Click-Plugins on Github](https://github.com/click-contrib/click-plugins)
[Click-Plugins on PyPi](https://pypi.python.org/pypi/click-plugins)


#### logging

## Performance TODO
* optimize data structures (numpy)
* use pypy
* employ @lrucache



app.py

cli.py and entry_point scripts 
The command line interface is implemented using the click project, which provides decorators for functions that become command line arguments.

Integration with setup.py entry_point is done via click-plugins, which allow cli commands to also be designated as entry point scripts.

See ccd.cli.py, setup.py and the click/click-plugin documentation.

DOC LINKS

logging 
Basic Python logging is used in pyccd and is fully configured in app.py. To use logging in any module:

from ccd import app
