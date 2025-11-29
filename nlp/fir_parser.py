# nlp/fir_parser.py
import re
from typing import Dict, List

def parse_fir_text(fir_text: str) -> Dict[str, List[str]]:
    """
    Parse FIR text into structured components.
    Returns victims, suspects, locations, times, and objects.
    """
    text = fir_text.lower()

    victims = re.findall(r'\b(victim|complainant|injured|deceased)\s+\w+', text)
    suspects = re.findall(r'\b(accused|suspect|offender)\s+\w+', text)
    locations = re.findall(r'\b(at|near|in)\s+[A-Z][a-z]+\b', fir_text)
    times = re.findall(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', fir_text)
    objects = re.findall(r'\b(knife|gun|weapon|blood|bag|phone)\b', text)

    return {
        "victims": victims,
        "suspects": suspects,
        "locations": locations,
        "times": times,
        "objects": objects
    }