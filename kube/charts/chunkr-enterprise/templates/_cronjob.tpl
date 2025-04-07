{{- define "chunkr.cronjob" -}}
{{- range $name, $service := .Values.services }}
{{- if and $service.enabled (eq ($service.type | default "deployment") "cronjob") }}
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: {{ $.Release.Name }}-{{ $name }}
  labels:
    app.kubernetes.io/name: {{ $name }}
    {{- include "chunkr.labels" $ | nindent 4 }}
spec:
  schedule: {{ $service.schedule | quote }}
  concurrencyPolicy: {{ $service.concurrencyPolicy | default "Forbid" }}
  successfulJobsHistoryLimit: {{ $service.successfulJobsHistoryLimit | default 3 }}
  failedJobsHistoryLimit: {{ $service.failedJobsHistoryLimit | default 1 }}
  jobTemplate:
    spec:
      template:
        metadata:
          labels:
            app.kubernetes.io/name: {{ $name }}
            {{- include "chunkr.labels" $ | nindent 12 }}
        spec:
          restartPolicy: {{ $service.podSpec.restartPolicy | default "Never" }}
          serviceAccountName: {{ $service.serviceAccountName | default (printf "%s-%s" $.Release.Name $name) }}
          {{- if $service.volumes }}
          volumes:
            {{- toYaml $service.volumes | nindent 12 }}
          {{- end }}
          containers:
          - name: {{ $name }}
            image: "{{ default $.Values.global.image.registry $service.image.registry }}/{{ $service.image.repository }}:{{ $service.image.tag }}"
            imagePullPolicy: {{ $.Values.global.image.pullPolicy }}
            {{- if $service.command }}
            command:
            {{- toYaml $service.command | nindent 14 }}
            {{- end }}
            {{- if $service.args }}
            args:
            {{- toYaml $service.args | nindent 14 }}
            {{- end }}
            {{- if or $service.useStandardEnv $service.envFrom }}
            envFrom:
            {{- if $service.useStandardEnv }}
            - configMapRef:
                name: {{ $.Release.Name }}-standard-env
            {{- end }}
            {{- if $service.envFrom }}
            {{- toYaml $service.envFrom | nindent 12 }}
            {{- end }}
            {{- end }}
            {{- if $service.env }}
            env:
            {{- tpl (toYaml $service.env) $ | nindent 14 }}
            {{- end }}
            {{- if $service.volumeMounts }}
            volumeMounts:
            {{- toYaml $service.volumeMounts | nindent 14 }}
            {{- end }}
            {{- if $service.resources }}
            resources:
              {{- toYaml $service.resources | nindent 16 }}
            {{- end }}
{{- end }}
{{- end }}
{{- end }}