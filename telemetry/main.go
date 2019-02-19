package main

import (
	"fmt"
	"github.com/influxdata/influxdb/client/v2"
	"github.com/jasonlvhit/gocron"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"

	httplogger "github.com/osunac/go-httplogger"
)

const (
	database      = "telemetry"
	dockerhub_url = "https://hub.docker.com/v2/repositories/loudml/community/"
	influxdb_url  = "http://localhost:8086"
	measurement   = "telemetry"
)

var db client.Client
var db_lock = &sync.Mutex{}

func influxDBClient() client.Client {
	c, err := client.NewHTTPClient(client.HTTPConfig{
		Addr: influxdb_url,
	})
	if err != nil {
		log.Fatalln("Error: ", err)
	}
	return c
}

/* Current timestamp in nanoseconds
 *
 * Documentation recommends to use the coarsest precision possible to improve
 * compression. Here we chose to round to seconds.
 *
 * https://docs.influxdata.com/influxdb/v1.7/write_protocols/line_protocol_tutorial/#timestamp
 */
func timestamp_ns() string {
	t := time.Now()
	t_s := strconv.FormatInt(t.Unix(), 10)
	t_ns := t_s + "000000"

	return t_ns
}

func route_api(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		var err error
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			panic(err)
		}
		m, err := metricsFromJSON(r.UserAgent(), body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusMethodNotAllowed)
			return
		}
		fmt.Fprintf(w, m.toInfluxDBLine(timestamp_ns(), "telemetry")+"\n")

		bp, _ := client.NewBatchPoints(client.BatchPointsConfig{
			Database:  database,
			Precision: "s",
		})

		pt, err := client.NewPoint(measurement, m.tags(), m.fields(), time.Now())
		if err != nil {
			log.Fatal(err, body)
		}
		bp.AddPoint(pt)

		db_lock.Lock()
		err = db.Write(bp)
		db_lock.Unlock()
		if err != nil {
			log.Fatal(err)
		}

	default:
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
	}
}

func collectDockerHubStats() {
	c := http.Client{}

	req, err := http.NewRequest("GET", dockerhub_url, nil)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := c.Do(req)
	if err != nil {
		log.Printf("%s", err)
		return
	}

	jsonBlob, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	i, err := imageInfoFromJSON(jsonBlob)
	if err != nil {
		log.Printf("error while parsing DockerHub JSON: %s", err)
		return
	}

	bp, _ := client.NewBatchPoints(client.BatchPointsConfig{
		Database:  database,
		Precision: "s",
	})

	pt, err := client.NewPoint(measurement, i.tags(), i.fields(), time.Now())
	if err != nil {
		log.Fatal(err)
	}
	bp.AddPoint(pt)

	db_lock.Lock()
	err = db.Write(bp)
	db_lock.Unlock()
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	db = influxDBClient()
	defer db.Close()

	collectDockerHubStats()
	gocron.Every(1).Hour().Do(collectDockerHubStats)
	<-gocron.Start()

	serveMux := http.NewServeMux()

	serveMux.HandleFunc("/api", route_api)
	srv := http.Server{
		Addr:    ":8080",
		Handler: httplogger.HTTPLogger(serveMux),
	}

	log.Fatal(srv.ListenAndServe())
}
