"""
LoudML fingerprints module
"""

from voluptuous import (
    All,
    Length,
    Range,
    Required,
)

from . import (
    schemas,
)
from .model import Model

class FingerprintsModel(Model):
    """
    Fingerprints model
    """

    TYPE = 'fingerprints'

    SCHEMA = Model.SCHEMA.extend({
        Required('term'): All(schemas.key, Length(max=256)),
        Required('max_terms'): All(int, Range(min=1)),
        Required('width'): All(int, Range(min=1)),
        Required('height'): All(int, Range(min=1)),
        'timestamp_field': schemas.key,
    })

    def __init__(self, settings, state=None):
        super().__init__(settings, state)

        self.term = settings['term']
        self.max_terms = settings['max_terms']
        self.w = settings['width']
        self.h = settings['height']
        self.timestamp_field = settings.get('timestamp_field', 'timestamp')
