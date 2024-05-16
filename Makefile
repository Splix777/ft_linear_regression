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

test:
	@echo "Running python tests"
	@. .venv/bin/activate; python3 -m unittest discover -s tests -p "*_test.py"

docker:
	@docker-compose -f docker-compose.yml up -d --build

status:
	@echo "\n\033[1;33mContainers\033[0m"
	@docker-compose ps
	@echo "\n\033[1;33mImages\033[0m"
	@docker-compose images
	@echo "\n\033[1;33mVolumes\033[0m"
	@docker volume ls
	@echo "\n\033[1;33mNetworks\033[0m"
	@docker network ls

fclean:
	docker-compose down
	docker-compose rm -f
	docker system prune -a -f
	docker volume prune -f
	docker network prune -f
	-docker volume ls -qf dangling=true | xargs -r docker volume rm
	rm -rf .venv

.PHONY: all python venv install run clean test