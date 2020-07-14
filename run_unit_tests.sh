rm .coverage
coverage run --branch -m unittest discover -s test -p '*Test.py'
coverage report -m
coverage html
