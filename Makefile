.PHONY: clean clean-test clean-pyc clean-build docs help lint checktypes checkstyle sast checklicenses test test-all coverage release virtual-environment

.DEFAULT_GOAL := help
PROJECT_NAME := $(shell poetry version | cut -d " " -f1 | tr '-' '_')
PROJECT_VERSION := $(shell poetry version -s)
SOURCES := src
TESTS := tests
REPORTS_DIR := .reports
VENV := $(shell poetry env info -p)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

version:
	@echo ${PROJECT_NAME} ${PROJECT_VERSION}

help:
	@poetry run python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: checktypes checkstyle sast checklicenses ## run all checks

lintprod: checktypes sast checklicenses ## run all checks

checktypes: virtual-environment ## check types with mypy
	poetry run mypy --ignore-missing-imports ${SOURCES} ${TESTS}

checkstyle: virtual-environment ## check style with flake8 and black
	poetry run pylint --rcfile=.pylintrc ${SOURCES} ${TESTS}
	poetry run flake8 --config=.flake8 ${SOURCES} ${TESTS}
	poetry run isort --check-only ${SOURCES} ${TESTS}
	poetry run black --check --diff ${SOURCES} ${TESTS}

fixstyle: virtual-environment ## fix black and isort style violations
	poetry run isort ${SOURCES} ${TESTS}
	poetry run black ${SOURCES} ${TESTS}

sast: virtual-environment ## run static application security testing
	poetry run bandit -r ${SOURCES}

checksafety: checkdependencies tmp-requirements.txt ## check dependencies meet licence rules
	poetry run safety check --full-report --ignore=40291

checkdependencies: $(REPORTS_DIR) virtual-environment ## check dependencies meet licence rules
	poetry run dependency-check --project ${PROJECT_NAME} --scan . -f JSON -f HTML -f XML --prettyPrint -o $(REPORTS_DIR)/ --exclude ".git/**" --exclude "$(VENV)/**" --exclude "**/__pycache__/**" --exclude "$(REPORTS_DIR)/**" --disableAssembly

checklicenses: virtual-environment tmp-requirements.txt ## check dependencies meet licence rules
	poetry run liccheck -s liccheck.ini -r tmp-requirements.txt

## run tests quickly with the default Python
## Multiple pytest runs are necessary as once the ${SOURCES} package has been loaded for a
## specific version of GSF, or with a custom shared object library, it cannot be unloaded.
test: virtual-environment
	poetry run pytest

test-all: virtual-environment ## run tests on every Python version with tox
	poetry run tox

show-coverage:
	xdg-open $(REPORTS_DIR)/htmlcov/index.html

$(REPORTS_DIR):
	mkdir -p $(REPORTS_DIR)

coverage-report: test $(REPORTS_DIR) ## check style with flake8 and black
	sed -re "s#<source>.*</source>#<source>.</source>#" -i $(REPORTS_DIR)/coverage.xml

flake8-report: $(REPORTS_DIR) virtual-environment ## check style with flake8 and black
	@rm $(REPORTS_DIR)/flake8-report.txt || echo 'No previous flake8 report found'
	poetry run flake8 --config=.flake8 --output-file $(REPORTS_DIR)/flake8-report.txt ${SOURCES} ${TESTS} || exit 0

bandit-report: $(REPORTS_DIR) virtual-environment ## run static application security testing
	@rm $(REPORTS_DIR)/bandit-report.json || echo 'No previous bandit report found'
	poetry run bandit -o $(REPORTS_DIR)/bandit-report.json -f json -r ${SOURCES} || exit 0

pylint-report: $(REPORTS_DIR) virtual-environment
	@rm $(REPORTS_DIR)/pylint-report.txt || echo 'No previous pylint report found'
	poetry run pylint --rcfile=.pylintrc --output-format=parseable ${SOURCES} ${TESTS} > $(REPORTS_DIR)/pylint-report.txt || exit 0

dependency-report: $(REPORTS_DIR) virtual-environment ## check dependencies meet licence rules
	poetry run dependency-check --project ${PROJECT_NAME} --scan . -f JSON -f HTML -f XML --prettyPrint -o $(REPORTS_DIR)/ --exclude ".git/**" --exclude "$(VENV)/**" --exclude "**/__pycache__/**" --exclude "$(REPORTS_DIR)/**" --disableAssembly

push-sonarqube: $(VENV)/bin/sonar-scanner  ## sonarqube scanner upload
	poetry run sonar-scanner \
		-Dsonar.qualitygate.wait=true -Dsonar.qualitygate.timeout=300 \
		-Dsonar.host.url=${SONARQUBE_URL} -Dsonar.login=${SONARQUBE_TOKEN} \
		-Dsonar.sources=poetry.lock,${SOURCES} -Dsonar.projectKey=${PROJECT_NAME} \
		-Dsonar.projectName=${PROJECT_NAME} -Dsonar.projectVersion=${PROJECT_VERSION} \
		-Dsonar.branch.name=${BRANCH}

sonarqube: pylint-report flake8-report bandit-report dependency-report coverage-report # push-sonarqube  ## sonarqube scanner upload

$(VENV)/bin/sonar-scanner: virtual-environment
	poetry run bin/install-sonar-scanner.py

release: ## package and upload a release
	test -z "$$(git status --untracked-files=no --porcelain .)"
	@make install
	@echo "Current version: '$$(poetry version) is going to $(BUMP)"
	poetry version $(BUMP)
	git commit -a -m "Bumped '$(BUMP)' version to $$(poetry version)"
	@make dist
	twine upload --skip-existing dist/*.whl
	@make tag
	git push

dist: clean ## builds source and wheel package
	poetry build

tmp-requirements.txt: poetry.lock
	poetry export --format requirements.txt --output tmp-requirements.txt
	@touch tmp-requirements.txt # when there are no dependencies

$(VENV): poetry.lock
	poetry update
	poetry install --remove-untracked

virtual-environment: $(VENV)

poetry.lock: pyproject.toml
	poetry lock --no-update

update:
	poetry install --remove-untracked
	poetry update
	@touch -c poetry.lock

install:
	poetry lock --no-update
	poetry install --extras pandas --remove-untracked

images:
	bin/build-images

pre-commit-install: virtual-environment
	poetry run pre-commit install

pre-commit-all: pre-commit-install
	poetry run pre-commit run --all-files

tag:
	git ls-remote --exit-code --tags origin ${TAG_NAME} || ( \
		git tag ${TAG_NAME} ; \
		git push origin ${TAG_NAME} ; \
	)

clean-git-merge:
	find . -name "*.py.orig" -o -name "*_BASE_*.py" -o -name "*_BACKUP_*.py" -o -name "*_LOCAL_*.py" -o -name "*_REMOTE_*.py" | xargs rm
