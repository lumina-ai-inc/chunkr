{{- define "chunkr.debug" -}}
{{- range $key, $value := .Values -}}
DEBUG: {{ $key }}: {{ $value | toYaml }}
{{ end -}}
{{- end -}} 