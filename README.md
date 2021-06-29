# Recycle It
> Recycle It helps users identify if something is recyclable or not.


[![Python Version][python-image]][python-url]

Recycle It helps users identify if something is recyclable or not.
Users upload an image and the system returns the information it found.

![](header.png)

## Installation

Don't forget to install geckodriver to run tests!
If on OS X Catalina (10.15) or above, refer to the instructions [here](https://firefox-source-docs.mozilla.org/testing/geckodriver/Notarization.html) to install.

OS X & Linux:

```sh
pip install -r requirements.txt
```

Windows:

```sh
pip install -r requirements.txt
```

## Usage example


_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

### Setting up workspace
Create new Conda Env.  
Pull from Remote / Main.  
**IMPORTANT** Ensure that you are always working on a branch, not MAIN.  
Check PATH. If geckodriver not found, copy it to one of the directions in PATH.
```sh
import sys

sys.path
````
Make sure to run migrations. In your local environment,
first run:
```sh
python recycle_it/manage.py makemigrations  
```
Then run: 
```sh
python recycle_it/manage.py migrate
```

### Tests
To run end-to-end testing run:
```sh
python tests/functional_tests.py
```

To run webapp tests:

```sh
python manage.py test homepage.tests
```
## Things to do when updating the project:
Don't forget to update the requirements.txt file.
Use pipreqs to update the requirements file.
```sh
pip install pipreqs

pipreqs --force --encoding=utf8
```

## Release History

* 0.0.1
    * Work in progress
    * ADD: Added `RecycleitFunctionalTest` for functional tests

## Meta

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/AGAN-Labs/recycle_it]()

## Contributing

1. Fork it (<https://github.com/AGAN-Labs/recycle_it/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

See ``RULES AND REGULATIONS`` for more information.

<!-- Markdown link & img dfn's -->
[python-image]: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
[python-url]: https://python.org/
[wiki]: https://github.com/AGAN-Labs/recycle_it/wiki


  