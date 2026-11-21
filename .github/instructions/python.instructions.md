---
description: "Use when writing or modifying Python code. Covers naming, typing, functions, conditions, formatting, and testing."
applyTo: "**/*.py"
---

# Python Guidelines

- Follow PEP 8 and write idiomatic Python.
- Use snake_case for functions, methods, variables, modules, and files.
- Use PascalCase for classes.
- Use UPPER_SNAKE_CASE for constants.
- Use meaningful names and avoid unnecessary abbreviations.
- Add type hints to public functions and methods.
- Prefer small, focused functions.
- Use pathlib for filesystem paths.
- Prefer direct boolean conditions over comparisons with True or False.
- Use early returns to reduce unnecessary nesting.
- Use standard library features when they are sufficient.
- Avoid mutable default arguments.
- Raise specific exceptions with useful error messages.
- Use context managers for resources that require cleanup.
- Follow existing project conventions.
- Name pytest test files test_*.py and test functions test_*.
- Keep tests deterministic and independent.
- Do not introduce dependencies without approval.