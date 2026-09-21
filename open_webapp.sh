#!/bin/bash
# open_webapp.sh
# Opens an authenticated local tunnel to the deployed Meridian webapp
# (Cloud Run, IAM-restricted to your Google account) and launches it in
# your default browser. Leave this terminal window open while using the
# app — closing it (or Ctrl+C) ends the tunnel and the app becomes
# unreachable until you run this again.
set -e
(sleep 2 && open http://127.0.0.1:8080) &
echo "Starting authenticated tunnel to meridian-web-ui — leave this open while using the app (Ctrl+C to stop)."
gcloud run services proxy meridian-web-ui --region us-central1 --project meridian-508414
