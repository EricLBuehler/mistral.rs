#!/bin/sh

if [ "X$TRT_API_BASE" = "X" ] ; then
  export TRT_API_BASE="http://127.0.0.1:3001/v1/"
fi

case "$1" in
  compl)
curl -X POST "${TRT_API_BASE}completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "stream": false,
  "prompt": "Here is a one line joke: ",
  "max_tokens": 100,
  "temperature": 0.7,
  "n": 1
}' | jq
;;

  stream)
curl -X POST "${TRT_API_BASE}completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "stream": true,
  "prompt": "Here is a one line joke: ",
  "max_tokens": 100,
  "temperature": 0.7,
  "n": 1
}'
;;

  chat)
curl -X POST "${TRT_API_BASE}chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Please tell me a one line joke."
    }
  ],
  "max_tokens": 100,
  "temperature": 1.2
}' | jq
;;

  json)
curl -v -X POST "${TRT_API_BASE}chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "seed": 42,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Please tell me a one line joke and rating."
    }
  ],
  "grammar": {
    "type": "json_schema",
    "value": {
      "type": "object",
      "properties": {
        "joke": {
          "type": "string"
        },
        "rating": {
          "type": "number"
        }
      },
      "additionalProperties": false,
      "required": ["joke", "rating"]
    }
  },
  "max_tokens": 100,
  "temperature": 1.2
}' | jq
;;

esac

echo
