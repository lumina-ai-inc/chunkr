{{- define "chunkr.job" -}}
{{- range $name, $service := .Values.services }}
{{- if and $service.enabled (eq ($service.type | default "deployment") "job") }}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ $.Release.Name }}-{{ $name }}
  labels:
    app.kubernetes.io/name: {{ $name }}
    {{- include "chunkr.labels" $ | nindent 4 }}
spec:
  {{- if $service.backoffLimit }}
  backoffLimit: {{ $service.backoffLimit }}
  {{- end }}
  {{- if $service.ttlSecondsAfterFinished }}
  ttlSecondsAfterFinished: {{ $service.ttlSecondsAfterFinished }}
  {{- end }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: {{ $name }}
        {{- include "chunkr.labels" $ | nindent 8 }}
    spec:
      restartPolicy: {{ $service.podSpec.restartPolicy | default "Never" }}
      {{- if $service.volumes }}
      volumes:
        {{- toYaml $service.volumes | nindent 8 }}
      {{- end }}
      containers:
      - name: {{ $name }}
        image: "{{ default $.Values.global.image.registry $service.image.registry }}/{{ $service.image.repository }}:{{ $service.image.tag }}"
        {{- if $service.command }}
        command:
        {{- toYaml $service.command | nindent 10 }}
        {{- end }}
        {{- if $service.env }}
        env:
        {{- tpl (toYaml $service.env) $ | nindent 10 }}
        {{- end }}
        {{- if $service.volumeMounts }}
        volumeMounts:
        {{- toYaml $service.volumeMounts | nindent 10 }}
        {{- end }}
        {{- if $service.resources }}
        resources:
          {{- toYaml $service.resources | nindent 12 }}
        {{- end }}
{{- end }}
{{- end }}
{{- end }} 