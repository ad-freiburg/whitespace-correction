.PHONY: checkstyle
checkstyle:
	@echo "Running flake8\n-----------------"
	flake8 tests
	flake8 trt
	@echo "\nRunning mypy\n-----------------"
	mypy tests
	mypy trt

.PHONY: tests
tests:
	pytest tests -n auto --disable-pytest-warnings
