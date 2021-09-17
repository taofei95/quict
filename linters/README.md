# Install isort.
pip install isort

# Automatically re-format your imports with isort.
isort --settings-path linters/tox.ini

# Install flake8.
pip install flake8

# Lint with flake8.
flake8 --config linters/tox.ini
