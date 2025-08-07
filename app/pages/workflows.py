"""Workflows page"""

import reflex as rx
from ..templates.template import template
from ..components.combobox import combobox


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    selected: str = ""
    selected_person: dict = {}
    query: str = ""
    
    people = [
        {"id": 1, "name": "Durward Reynolds"},
        {"id": 2, "name": "Kenton Towne"},
        {"id": 3, "name": "Therese Wunsch"},
        {"id": 4, "name": "Benedict Kessler"},
        {"id": 5, "name": "Katelyn Rohan"},
    ]

    def set_selected(self, value: str) -> None:
        """Set selected value."""
        self.selected = value
        
    def set_selected_person(self, person) -> None:
        """Set selected person."""
        print(f"Selected person called with: {person}, type: {type(person)}")  # Debug logging
        
        if person is None:
            # Handle clearing selection
            self.selected_person = {}
            self.query = ""
            print("Cleared selection")
        elif isinstance(person, dict):
            # Handle dict selection (expected case)
            self.selected_person = person
            if "name" in person:
                self.query = person["name"]
                print(f"Updated query to show selection: {self.query}")
        elif isinstance(person, str):
            # Handle string selection (fallback - try to find matching person)
            matching_person = None
            for p in self.people:
                if p["name"] == person:
                    matching_person = p
                    break
            
            if matching_person:
                self.selected_person = matching_person
                self.query = person
                print(f"Found matching person: {matching_person}")
            else:
                # If no match found, create a simple dict
                self.selected_person = {"name": person}
                self.query = person
                print(f"Created person from string: {self.selected_person}")
        else:
            # Handle other types
            self.selected_person = {"name": str(person)}
            self.query = str(person)
            print(f"Converted to person: {self.selected_person}")
            
        print(f"Selected person set to: {self.selected_person}")
        
    def set_query(self, query: str) -> None:
        """Set search query."""
        print(f"Query changed to: '{query}'")
        self.query = query
        # Clear selection when user starts typing to allow free text input
        if self.selected_person:
            print("Clearing selection to allow typing")
            self.selected_person = {}
        
    @rx.var
    def filtered_people(self) -> list[dict]:
        """Get filtered people based on query."""
        # Always filter based on current query
        if not self.query:
            return self.people
        return [p for p in self.people if self.query.lower() in p["name"].lower()]
        
    @rx.var
    def display_value(self) -> str:
        """Get display value for combobox input."""
        # Always use the query for display to allow editing
        return self.query


@rx.page(route="/workflows")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The workflows page."""
    return rx.vstack(
        rx.heading("Workflows", size="9"),
        rx.text(f"Selected: {State.selected}"),
        rx.text(f"Selected person: {State.selected_person}"),
        
        # Component comparison
        rx.vstack(
            rx.heading("Component Comparison", size="6", margin_bottom="4"),
            
            # Reflex Select for comparison
            rx.box(
                rx.text("Reflex Select:", font_weight="bold", margin_bottom="2"),
                rx.select(
                    ["Durward Reynolds", "Kenton Towne", "Therese Wunsch", "Benedict Kessler", "Katelyn Rohan"],
                    placeholder="Select a person...",
                    on_change=State.set_selected,
                    value=State.selected,
                    size="2"
                ),
                class_name="mb-6 max-w-xs"
            ),
            
            # Our styled Combobox
            rx.box(
                rx.text("HeadlessUI Combobox:", font_weight="bold", margin_bottom="2"),
                combobox(
                    options=State.filtered_people,
                    on_change=State.set_selected_person,
                    on_input_change=State.set_query,
                    input_value=State.display_value,
                    placeholder="Search people...",
                    display_field="name"
                ),
                class_name="mb-6 max-w-xs"
            ),
            
            class_name="mt-8"
        ),
        
        spacing="5",
        justify="center",
        min_height="85vh",
    )
