"""
Elasticsearch module for Loud ML
for connecting to Amazon Elasticsearch Service
"""

import datetime
import logging

import elasticsearch.exceptions
import urllib3.exceptions

from elasticsearch import (
    Elasticsearch,
    TransportError,
    RequestsHttpConnection,
)

from requests_aws4auth import AWS4Auth
import boto3

from voluptuous import (
    Required,
    Optional,
    All,
    Length,
    Boolean,
    Schema,
)

from . import (
    errors,
    schemas,
)

from loudml.datasource import DataSource
from loudml.elastic import ElasticsearchDataSource 

class ElasticsearchAWSDataSource(ElasticsearchDataSource):
    """
    Elasticsearch datasource on AWS
    Documentation: https://docs.aws.amazon.com/elasticsearch-service/latest/developerguide/es-indexing-programmatic.html
    """

    SCHEMA = DataSource.SCHEMA.extend({
        # eg, my-test-domain.us-east-1.es.amazonaws.com
        Required('host'): str,
        Required('region'): str,
        Required('index'): str,
        'routing': str,
        Optional('access_key'): All(schemas.key, Length(max=256)),
        Optional('secret_key'): str,
        Optional('get_boto_credentials', default=False): Boolean(),
    })

    def __init__(self, cfg):
        super().__init__(cfg)
        cfg['type'] = 'elasticsearch_aws'

    @property
    def host(self):
        return self.cfg['host']

    @property
    def region(self):
        return self.cfg['region']

    @property
    def aws_access_key(self):
        return self.cfg.get('access_key')

    @property
    def aws_secret_key(self):
        return self.cfg.get('secret_key')

    @property
    def get_boto_credentials(self):
        return self.cfg.get('get_boto_credentials') or False

    @property
    def es(self):
        if self._es is None:
            logging.info('connecting to elasticsearch on AWS %s:443 region:%s',
                         self.host, self.region)

            service = 'es'
            if self.get_boto_credentials:
                credentials = boto3.Session().get_credentials()
                awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, self.region, service)
            elif not (self.aws_access_key is None or self.aws_secret_key is None):
                awsauth = AWS4Auth(self.aws_access_key, self.aws_secret_key, self.region, service)
            else:
                exn = 'invalid configuration: AWS credentials not found'
                raise errors.DataSourceError(self.name, exn)

            self._es = Elasticsearch(
                hosts = [{'host': self.host, 'port': 443}],
                http_auth = awsauth,
                use_ssl = True,
                verify_certs = True,
                connection_class = RequestsHttpConnection
            )


        # urllib3 & elasticsearch modules log exceptions, even if they are
        # caught! Disable this.
        urllib_logger = logging.getLogger('urllib3')
        urllib_logger.setLevel(logging.CRITICAL)
        es_logger = logging.getLogger('elasticsearch')
        es_logger.setLevel(logging.CRITICAL)

        return self._es

