"""
Reflex custom component Querybuilder.

Based on https://react-querybuilder.js.org
"""

# For wrapping react guide, visit https://reflex.dev/docs/wrapping-react/overview/

import reflex as rx
from dataclasses import dataclass
from dataclasses_json import dataclass_json


# Some libraries you want to wrap may require dynamic imports.
# This is because they they may not be compatible with Server-Side Rendering (SSR).
# To handle this in Reflex, all you need to do is subclass `NoSSRComponent` instead.
# For example:
# from reflex.components.component import NoSSRComponent
# class Querybuilder(NoSSRComponent):
#     pass
@dataclass_json
@dataclass
class Field:
    name: str
    label: str


class Querybuilder(rx.Component):
    """Querybuilder component."""

    # The React library to wrap.
    library = "react-querybuilder"

    # The React component tag.
    tag = "QueryBuilder"

    # If the tag is the default export from the module, you must set is_default = True.
    # This is normally used when components don't have curly braces around them when importing.
    # is_default = True

    # If you are wrapping another components with the same tag as a component in your project
    # you can use aliases to differentiate between them and avoid naming conflicts.
    # alias = "OtherQuerybuilder"

    # The props of the React component.
    # Note: when Reflex compiles the component to Javascript,
    # `snake_case` property names are automatically formatted as `camelCase`.
    # The prop names may be defined in `camelCase` as well.
    # some_prop: rx.Var[str] = "some default value"
    # some_other_prop: rx.Var[int] = 1

    # By default Reflex will install the library you have specified in the library property.
    # However, sometimes you may need to install other libraries to use a component.
    # In this case you can use the lib_dependencies property to specify other libraries to install.
    # lib_dependencies: list[str] = []

    # Event triggers declaration if any.
    # Below is equivalent to merging `{ "on_change": lambda e: [e] }`
    # onto the default event triggers of parent/base Component.
    # The function defined for the `on_change` trigger maps event for the javascript
    # trigger to what will be passed to the backend event handler function.
    # on_change: rx.EventHandler[lambda e: [e]]
    fields: rx.Var[list[Field]]

    # TODO: do we need to implement `query` and on_query_change as in
    # https://codesandbox.io/p/sandbox/github/react-querybuilder/react-querybuilder/tree/
    # main/examples/basic-ts?file=%2Fsrc%2FApp.tsx%3A6%2C25-9%2C3

    # To add custom code to your component
    # def _get_custom_code(self) -> str:
    #     return "const customCode = 'customCode';"
    def add_imports(self):
        return {"": ["react-querybuilder/dist/query-builder.css"]}


querybuilder = Querybuilder.create
