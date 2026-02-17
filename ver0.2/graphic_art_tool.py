def graphic_art(prompt):
    """Generate graphic art based on a prompt."""
    return {"image_url": f"https://example.com/{prompt.replace(' ', '_')}.png"}


schema = {
    "type": "function",
    "function": {
        "name": "graphic_art",
        "description": "Generate graphic art based on a prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt describing the desired graphic art."
                }
            },
            "required": ["prompt"]
        }
    }
}
