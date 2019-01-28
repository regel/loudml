package main

import (
	"fmt"
	"github.com/influxdata/influxdb/client/v2"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"time"

	httplogger "github.com/osunac/go-httplogger"
)

const (
	database     = "telemetry"
	influxdb_url = "http://localhost:8086"
	measurement  = "telemetry"
)

var db client.Client

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
		m, err := metricsFromJSON(body)
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
			log.Fatal(err)
		}
		bp.AddPoint(pt)

		err = db.Write(bp)
		if err != nil {
			log.Fatal(err)
		}

	default:
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
	}
}

func main() {
	db = influxDBClient()
	defer db.Close()

	serveMux := http.NewServeMux()

	serveMux.HandleFunc("/api", route_api)
	srv := http.Server{
		Addr:    ":8080",
		Handler: httplogger.HTTPLogger(serveMux),
	}

	log.Fatal(srv.ListenAndServe())
}
