job "loudml" {
  datacenters = ["dc1"]
  constraint {
    attribute = "${attr.cpu.arch}"
    value = "amd64"
	}
	update {
    stagger = "10s"
    max_parallel = 1
	}

	group "loudml" {
    restart {
      attempts = 2
      interval = "1m"
      delay = "10s"
      mode = "fail"
    }

    task "loudml" {
      driver = "docker"
      config {
        image = "loudml/community"
        hostname = "loudml"
        port_map {
          loudml = 8077
        }
        volumes = ["local/var/loudml:/var/lib/loudml", "local/loudml.yml:/etc/loudml/config.yml"]
      }

      service {
        name = "loudml"
        tags = ["global", "ml", "urlprefix-loudml.service.consul:9999/"]
        port = "loudml"
        check {
          name = "alive"
          type = "tcp"
          interval = "10s"
          timeout = "2s"
        }
      }
      template {
        data = <<EOTC
---
datasources:
  - name: kapacitor
    type: influxdb
    addr: kapacitor.service.consul:9092
    database: loudml
    retention_policy: autogen
    create_database: false
  - name: influx
    type: influxdb
    addr: influxdb.service.consul:8086
    database: telegraf
    retention_policy: autogen

storage:
  path: /var/lib/loudml

server:
  listen: 0.0.0.0:8077
EOTC
        destination = "local/loudml.yml"
      }
      resources {
        cpu = 2000 # 500 MHz
        memory = 8096 # 256MB
        network {
          mbits = 10
          port "loudml" {
            static = "8077"
          }
        }
      }
    }
    task "influxdb" {
      driver = "docker"
      config {
        image = "influxdb:alpine"
        hostname = "influxdb"
        port_map {
          http = 8086
        }
      }

      resources {
        cpu    = 500
        memory = 500
        network {
          mbits = 10
          port "http" {
            static = "8086"
          }
        }
      }

      service {
			  tags = ["global", "ml", "urlprefix-influxdb.service.consul:9999/"]
        name = "influxdb"
        port = "http"
        check {
          name     = "InfluxDB HTTP"
          type     = "http"
          path     = "/ping"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
    task "chronograf" {
      driver = "docker"
      config {
        image = "loudml/chronograf"
        hostname = "chronograf"
        port_map {
          chronograf = 8888
        }
        volumes = ["local/var/chronograf:/var/lib/chronograf"]
      }
      env {
        INFLUXDB_URL = "http://influxdb.service.consul:8086"
        LOUDML_URL = "http://loudml.service.consul:8077"
        KAPACITOR_URL = "http://kapacitor.service.consul:9092"
      }
      service {
        name = "chronograf"
        tags = ["global", "ml", "urlprefix-chronograf.service.consul:9999/"]
        port = "chronograf"
        check {
          name = "alive"
          type = "tcp"
          interval = "10s"
          timeout = "2s"
        }
      }
      resources {
        cpu = 500 # 500 MHz
        memory = 1024 # 256MB
        network {
          mbits = 10
          port "chronograf" {
            static = "8888"
          }
        }
      }
    }
    task "kapacitor" {
      driver = "docker"
      config {
        image = "kapacitor:latest"
        hostname = "kapacitor"
        port_map {
          kapacitor = 9092
        }
        volumes = ["local/kapacitor:/var/lib/kapacitor","local/kapacitor.conf:/etc/kapacitor/kapacitor.conf"]
      }
      env {
        KAPACITOR_INFLUXDB_0_URLS_0 = "http://influxdb.service.consul:8086"
      }
      service {
        name = "kapacitor"
        tags = ["global", "ml", "urlprefix-kapacitor.service.consul:9999/"]
        port = "kapacitor"
        check {
          name = "alive"
          type = "tcp"
          interval = "10s"
          timeout = "2s"
        }
      }

      template {
        data = <<EOTC
data_dir = "/var/lib/kapacitor"

[http]
    bind-address = "0.0.0.0:9092"
    log-enabled = true
    write-tracing = false
    pprof-enabled = false
    https-enabled = false

[logging]
    # Destination for logs
    # Can be a path to a file or 'STDOUT', 'STDERR'.
    file = "/var/log/kapacitor/kapacitor.log"
    # Logging level can be one of:
    # DEBUG, INFO, ERROR
    # HTTP logging can be disabled in the [http] config section.
    level = "INFO"

[replay]
  dir = "/var/lib/kapacitor/replay"

[storage]
  boltdb = "/var/lib/kapacitor/kapacitor.db"
EOTC
        destination = "local/kapacitor.conf"
      }

      resources {
        cpu = 500 # 500 MHz
        memory = 1024 # 256MB
        network {
          mbits = 10
          port "kapacitor" {
            static = "9092"
          }
        }
      }
    }
  }
}
