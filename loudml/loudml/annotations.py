import logging
import uuid

from loudml.api import (
    Hook,
)

from voluptuous import (
    ALLOW_EXTRA,
    All,
    Any,
    Optional,
    Required,
    Schema,
)

class AnnotationHook(Hook):
    CONFIG_SCHEMA = Schema({
        Required('type'): str,
        Optional('text'): str,
# TODO: Add tags
    }, extra=ALLOW_EXTRA)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = self.config.get('text', '{desc}')
        assert self.source

    def on_anomaly_start(
        self,
        model,
        dt,
        score,
        predicted,
        observed,
        anomalies,
        *args,
        **kwargs
    ):
        # Deal with anomaly notification here
        ano_desc = [
            "feature '{}' is too {} (score = {:.1f})".format(
                 feature,
                 ano['type'],
                 ano['score']
            )
            for feature, ano in anomalies.items()
        ]
        _desc = self.text.format(desc="; ".join(ano_desc))
        _id = str(uuid.uuid4())
        points = self.source.insert_annotation(
            dt,
            _desc,
            self.config['type'],
            _id,
        )
        self.set_object('annotations.points', points)

    def on_anomaly_end(
        self,
        model,
        dt,
        score,
        *args,
        **kwargs
    ):
        try:
            points = self.get_object('annotations.points')
        except KeyError:
            return

        self.source.update_annotation(dt, points)
        self.delete_object('annotations.points')
