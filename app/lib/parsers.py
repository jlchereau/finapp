"""
Parsers
"""

from typing import Any, Dict, Optional, Type
import jmespath
from pydantic import BaseModel, create_model, Field

# Global cache (model name → generated model)
MODEL_CACHE: Dict[str, Type[BaseModel]] = {}


class PydanticJSONParser:
    """
    Parser for JSON data using Pydantic models.
    The model is generated dynamically based on the provided configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config["name"]
        self.fields_config = config.get("fields", {})
        self.model = self._get_or_create_model()

    def _get_or_create_model(self) -> Type[BaseModel]:
        """
        Get or create the Pydantic model.
        """
        if self.model_name in MODEL_CACHE:
            return MODEL_CACHE[self.model_name]
        fields = {}
        for field_name, conf in self.fields_config.items():
            default_value = conf.get("default", None)
            fields[field_name] = (Optional[Any], Field(default=default_value))
        model = create_model(self.model_name, **fields)
        MODEL_CACHE[self.model_name] = model
        return model

    def create_instance(self, json_data: Dict[str, Any]) -> BaseModel:
        """
        Create an instance of the Pydantic model from the provided JSON data.
        """
        if not json_data:
            return self.model()
        extracted_data = {}
        for key, conf in self.fields_config.items():
            expr = conf.get("expr")
            default = conf.get("default")
            value = jmespath.search(expr, json_data)
            if value is None:
                value = default
            extracted_data[key] = value
        return self.model(**extracted_data)
