from pydantic import BaseModel
from typing import Optional, List, Union, Type
import json

class Property(BaseModel):
    name: str
    prop_type: str
    description: Optional[str] = None
    default: Optional[str] = None

class JsonSchema(BaseModel):
    title: str
    properties: List[Property]

def from_pydantic(pydantic: Union[BaseModel, Type[BaseModel]], current_depth: int = 0) -> dict:
    """Convert a Pydantic model to a Chunk json schema."""
    MAX_DEPTH = 5
    model = pydantic if isinstance(pydantic, type) else pydantic.__class__
    schema = model.model_json_schema()
    properties = []

    def get_enum_description(details: dict) -> str:
        """Get description including enum values if they exist"""
        description = details.get('description', '')
        
        # First check if this is a direct enum
        if 'enum' in details:
            enum_values = details['enum']
            enum_str = '\nAllowed values:\n' + '\n'.join(f'- {val}' for val in enum_values)
            return f"{description}{enum_str}"
            
        # Then check if it's a reference to an enum
        if '$ref' in details:
            ref_schema = resolve_ref(details['$ref'], schema.get('$defs', {}))
            if 'enum' in ref_schema:
                enum_values = ref_schema['enum']
                enum_str = '\nAllowed values:\n' + '\n'.join(f'- {val}' for val in enum_values)
                return f"{description}{enum_str}"
                
        return description

    def resolve_ref(ref: str, definitions: dict) -> dict:
        """Resolve a $ref reference to its actual schema"""
        if not ref.startswith('#/$defs/'):
            return {}
        ref_name = ref[len('#/$defs/'):]
        return definitions.get(ref_name, {})

    def get_nested_schema(field_schema: dict, depth: int) -> dict:
        if depth >= MAX_DEPTH:
            return {}
        
        # If there's a $ref, resolve it first
        if '$ref' in field_schema:
            field_schema = resolve_ref(field_schema['$ref'], schema.get('$defs', {}))
        
        nested_props = {}
        if field_schema.get('type') == 'object':
            for name, details in field_schema.get('properties', {}).items():
                if details.get('type') == 'object' or '$ref' in details:
                    ref_schema = details
                    if '$ref' in details:
                        ref_schema = resolve_ref(details['$ref'], schema.get('$defs', {}))
                    nested_schema = get_nested_schema(ref_schema, depth + 1)
                    nested_props[name] = {
                        'type': 'object',
                        'description': get_enum_description(details),
                        'properties': nested_schema
                    }
                else:
                    nested_props[name] = {
                        'type': details.get('type', 'string'),
                        'description': get_enum_description(details)
                    }
        return nested_props

    for name, details in schema.get('properties', {}).items():
        # Handle arrays
        if details.get('type') == 'array':
            items = details.get('items', {})
            if '$ref' in items:
                items = resolve_ref(items['$ref'], schema.get('$defs', {}))
            
            # Get nested schema for array items
            item_schema = get_nested_schema(items, current_depth)
            description = get_enum_description(details)
            
            if item_schema:
                description = f"{description}\nList items schema:\n{json.dumps(item_schema, indent=2)}"
            
            prop = Property(
                name=name,
                prop_type='list',
                description=description
            )
        # Handle objects and references
        elif details.get('type') == 'object' or '$ref' in details:
            prop_type = 'object'
            ref_schema = details
            if '$ref' in details:
                ref_schema = resolve_ref(details['$ref'], schema.get('$defs', {}))
            
            nested_schema = get_nested_schema(ref_schema, current_depth)
            
            prop = Property(
                name=name,
                prop_type=prop_type,
                description=get_enum_description(details),
                properties=nested_schema
            )
            
        # Handle primitive types
        else:
            prop = Property(
                name=name,
                prop_type=details.get('type', 'string'),
                description=get_enum_description(details),
                default=str(details.get('default')) if details.get('default') is not None else None
            )
            
        properties.append(prop)
    
    json_schema = JsonSchema(
        title=schema.get('title', model.__name__),
        properties=properties
    )
    
    return json_schema.model_dump(mode="json", exclude_none=True)