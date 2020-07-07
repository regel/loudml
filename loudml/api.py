"""
Loud ML hook API
"""

import os
import yaml


def validate(schema, data):
    """
    Validate data against a schema
    """
    return schema(data) if schema else data


class Plugin:
    """
    Base class for LoudML plug-in
    """

    CONFIG_SCHEMA = None
    instance = None

    def __init__(self, name, config_dir):
        self.set_instance(self)
        self.name = name
        self.config = None
        path = os.path.join(config_dir, 'plugins.d', '{}.yml'.format(name))

        try:
            with open(path) as config_file:
                config = yaml.load(config_file)
        except FileNotFoundError:
            # XXX No config or plug-in disabled
            return

        self.config = self.validate(config)

    @classmethod
    def validate(cls, config):
        """
        Validate plug-in configuration
        """
        return validate(cls.CONFIG_SCHEMA, config)

    @classmethod
    def set_instance(cls, instance):
        cls.instance = instance


class Hook:
    """
    Generate notification
    """
    CONFIG_SCHEMA = None

    def __init__(
        self,
        name,
        config,
        model,
        storage,
        source=None,
        *args,
        **kargs
    ):
        self.name = name
        self.model = model
        self.storage = storage
        self.source = source
        self.config = self.validate(config)

        # Index features by name
        self.features = {}
        features = self.model.get('features')

        if isinstance(features, dict):
            features = features.get('i', []) \
                + features.get('o', []) \
                + features.get('io', [])

        for feature in features:
            self.features[feature['name']] = feature

    def feature_to_str(self, name):
        """
        Return string describing the feature
        """
        feature = self.features.get(name)
        if not feature:
            return name

        match_all = feature.get('match_all', [])
        if len(match_all) == 0:
            return name

        tags = ",".join([
            "{}:{}".format(tag['tag'], tag['value'])
            for tag in match_all
        ])
        return "{}[{}]".format(name, tags)

    @classmethod
    def validate(cls, config):
        """
        Validate hook configuration
        """
        return validate(cls.CONFIG_SCHEMA, config)

    def on_anomaly_start(
        self,
        dt,
        score,
        predicted,
        observed,
        anomalies,
        *args,
        **kwargs
    ):
        """
        Callback function called on anomaly detection

        dt -- Timestamp of the anomaly as a datetime.datetime object
        score -- Computed anomaly score [0-100]
        predicted -- Predicted values
        observed -- Observed values
        mse -- MSE
        anomalies -- dict of abnormal features
        """
        raise NotImplementedError()

    def on_anomaly_end(self, dt, score, *args, **kwargs):
        """
        Callback function called on anomaly detection

        dt -- Timestamp of the anomaly as a datetime.datetime object
        score -- Computed anomaly score [0-100]
        """
        pass

    def set_object(self, key, data):
        """
        Save a persistent object

        Useful to keep persistent data accross hook calls.

        key -- object identifier
        data -- jsonifiable data
        """

        self.storage.set_model_object(self.model['name'], key, data)

    def get_object(self, key):
        """
        Get a persistent object

        key -- object identifier
        """

        return self.storage.get_model_object(self.model['name'], key)

    def delete_object(self, key):
        """
        Delete a persistent object

        key -- object identifier
        """

        return self.storage.delete_model_object(self.model['name'], key)
