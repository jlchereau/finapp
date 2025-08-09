Create or improve Finapp components

$ARGUMENTS

IMPORTANT:
- We are developping for python v3.10+ (modernize code) and we work from a vitual environment in @.venv/.
- Reflex v0.8+ code can be found in @.venv/lib/python3.12/site-packages/reflex/ and documentation and sample code at https://reflex.dev/docs/.
- This command is used in the scope of the @app/components/ and @tests/components folders (request permission to work outside the scope if needed).
- Make sure to maintain test coverage in line with component code.
- Once you have completed your changes, proceed in the following order:
    1. Call `make format` to format code with black.
    2. Call `make lint` to check for linting errors with pylint and flake8.
    3. Fix linting errors (ignore the errors outside the scope) but avoid adding ignore/disable comments (e.g. # pylint: disable=inherit-non-class) unlesss there is no other option (include a python comment to justify why).
    4. Use `make build` to check for compilation errors and fix them.
    5. Since `make test` is slow, first use pytest from .venv to limit tests to your changes then check regressions with `make test`.
    6. Iterate from 1 to 6 as you make changes, until all issues are resolved.
- Never call `make run` which is a blocking command. Before you intend to run the playright MCP server, please ask me to call `make run` in a terminal for you. 
