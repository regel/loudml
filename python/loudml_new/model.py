"""
LoudML model
"""

class Model:
    """
    LoudML model
    """

    def __init__(self, name, data):
        """
        name -- model name
        data -- model data
        """

        self.name = name
        self.data = data
        self.index = data['index']
        self.routing = data.get('routing')
        self.data['name'] = name
        self.state = None

    @property
    def features(self):
        """Model features"""
        return self.data['features']
