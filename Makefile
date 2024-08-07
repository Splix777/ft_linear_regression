all:
	@echo "Usage: make python|venv|install|run|clean|test"

python: venv install

venv:
	@echo "Creating python virtual environment"
	@python3 -m venv .venv

install:
	@echo "Installing python packages"
	@. .venv/bin/activate; pip install -r requirements.txt

run: venv install
	@echo "Running python script"
	@. .venv/bin/activate; python3 main.py

clean:
	@echo "Cleaning python virtual environment"
	@rm -rf .venv

.PHONY: all python venv install run clean