
# install default libraries for python
default:
	pip install lint &&
		pip install black &&
			pip install pylint

# install libraries cmd
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# install requirments.txt
freeze:
	pip freeze > requirements.txt

# run test code
test:
	python test.py

# format code
format:
	black *.py
lint:
	pylint --disable=R,C file.py

run:
	#test
	python main.py
deploy:
	#deploy
all:
	install lint test deploy