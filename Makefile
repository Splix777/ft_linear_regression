all:
	@echo "Usage: make python|venv|install|run|clean|test|fclean"

python: venv install

venv:
	@echo "Creating python virtual environment"
	@python3 -m venv .venv

install:
	@echo "Installing python packages"
	@. .venv/bin/activate; pip install -r requirements.txt

run:
	@echo "Running python script"
	@. .venv/bin/activate; python3 main.py

clean:
	@echo "Cleaning python virtual environment"
	@rm -rf .venv

test:
	@echo "Running python tests"
	@. .venv/bin/activate; python3 -m unittest discover -s tests -p "*_test.py"

fclean: clean
	@echo "Cleaning python cache"
	@find . -type d -name __pycache__ -exec rm -rf {} \;
	@find . -type f -name '*.pyc' -exec rm -f {} \;

.PHONY: all python venv install run clean test fclean