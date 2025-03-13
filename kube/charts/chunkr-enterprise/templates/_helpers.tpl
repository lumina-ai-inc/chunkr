{{/*
Common labels
*/}}
{{- define "chunkr.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
{{/*
Selector labels
*/}}
{{- define "chunkr.selectorLabels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Standard environment variables
Merges common environment variables with service-specific ones
Usage: include "chunkr.standardEnv" (dict "Values" .Values "serviceEnv" .serviceEnv)
*/}}
{{- define "chunkr.standardEnv" -}}
{{- if .Values -}}
  {{- if .Values.common -}}
    {{- $standardEnv := .Values.common.standardEnv | default list -}}
    {{- $serviceEnv := .serviceEnv | default list -}}
    {{- concat $standardEnv $serviceEnv | toYaml -}}
  {{- else -}}
    {{- $serviceEnv := .serviceEnv | default list -}}
    {{- toYaml $serviceEnv -}}
  {{- end -}}
{{- else -}}
  {{- fail "Values is nil in standardEnv template" -}}
{{- end -}}
{{- end -}}
