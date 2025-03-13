{{- define "chunkr.service" -}}
{{- range $name, $service := .Values.services }}
{{- if and $service.enabled $service.port }}
---
apiVersion: v1
kind: Service
metadata:
  name: {{ $.Release.Name }}-{{ $name }}
  labels:
    app.kubernetes.io/name: {{ $name }}
    {{- include "chunkr.labels" $ | nindent 4 }}
spec:
  type: ClusterIP
  ports:
    - port: {{ $service.port }}
      targetPort: {{ $service.targetPort | default $service.port }}
      protocol: TCP
      name: http
  selector:
    app.kubernetes.io/name: {{ $name }}
{{- end }}
{{- end }}
{{- end -}}