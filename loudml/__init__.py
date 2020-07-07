from collections import defaultdict
from loudml.donut import DonutModel
from loudml.annotations import AnnotationHook
from loudml.influx import InfluxBucket
from loudml.elastic import ElasticsearchBucket
from loudml.elastic_aws import ElasticsearchAWSBucket
from loudml.warp10 import Warp10Bucket
from loudml.mongo import MongoBucket
from loudml.opentsdb import OpenTSDBBucket
from loudml.prometheus import PrometheusBucket


entry_points = defaultdict(list, {
    'loudml.models': [
        ('donut', DonutModel),
    ],
    'loudml.hooks': [
        ('annotations', AnnotationHook),
    ],
    'loudml.buckets': [
        ('influxdb', InfluxBucket),
        ('elasticsearch', ElasticsearchBucket),
        ('elasticsearch_aws', ElasticsearchAWSBucket),
        ('warp10', Warp10Bucket),
        ('mongodb', MongoBucket),
        ('opentsdb', OpenTSDBBucket),
        ('prometheus', PrometheusBucket),
    ],
})


def load_entry_point(namespace, name):
    """
    Load entry point
    """
    global entry_points
    for ep in entry_points[namespace]:
        if ep[0] == name:
            cls = ep[1]
            return cls
    return None
