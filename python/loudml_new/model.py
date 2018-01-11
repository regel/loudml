"""
LoudML model
"""

class Model:
    """
    LoudML model
    """

    def __init__(self, settings, state=None):
        """
        name -- model settings
        """

        self.name = settings.get('name')
        self._settings = settings
        self.index = settings.get('index')
        self.db = settings.get('db')
        self.measurement = settings.get('measurement')
        self.routing = settings.get('routing')
        self.state = state

    @property
    def features(self):
        """Model features"""
        return self._settings['features']

    @property
    def settings(self):
        return self._settings

    @property
    def data(self):
        return {
            'settings': self.settings,
            'state': self.state,
        }
