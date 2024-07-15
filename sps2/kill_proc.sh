#! /usr/bin/env bash

kill -9 $(ps -ef|grep "spawn"|grep -v grep|awk '{print $2}')
kill -9 $(ps -ef|grep "mlsa_w"|grep -v grep|awk '{print $2}')
