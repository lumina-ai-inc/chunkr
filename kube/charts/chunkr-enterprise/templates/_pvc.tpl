{{- define "chunkr.pvc" -}}
{{- range $name, $service := .Values.services }}
{{- if and $service.enabled $service.persistence (and $service.persistence.enabled) }}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ $service.persistence.name }}
  labels:
    {{- include "chunkr.labels" $ | nindent 4 }}
  {{- with $service.persistence.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  accessModes:
    {{- toYaml $service.persistence.accessModes | nindent 4 }}
  resources:
    requests:
      storage: {{ $service.persistence.size }}
  storageClassName: {{ default "standard" $.Values.global.storageClass }}
{{- end }}
{{- end }}
{{- end }} 
