package utils

import (
	"encoding/json"

	"github.com/sirupsen/logrus"
)

func init() {
	logrus.SetFormatter(&logrus.TextFormatter{TimestampFormat: "2006-01-02 15:04:05.000", FullTimestamp: true})
}

func PanicError(err error) {
	if err != nil {
		logrus.Panic(err)
	}
}

func JsonifyNested(v interface{}) string {
	d, err := json.Marshal(v)
	if err != nil {
		logrus.Warnf("Failed to jsonify nested object: %v", err)
		return ""
	}
	return string(d)
}
