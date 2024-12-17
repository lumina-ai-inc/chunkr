{{- define "chunkr.deployment" -}}
{{- range $name, $service := .Values.services }}
{{- if $service.enabled }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $.Release.Name }}-{{ $name }}
  labels:
    app.kubernetes.io/name: {{ $name }}
    {{- include "chunkr.labels" $ | nindent 4 }}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: {{ $name }}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: {{ $name }}
    spec:
      {{- if $service.useGPU }}
      affinity:
        {{- $gpuAffinity := deepCopy $.Values.global.gpuWorkload.affinity }}
        {{- $matchExpr := index $gpuAffinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution 0 -}}
        {{- $matchExpressions := $matchExpr.podAffinityTerm.labelSelector.matchExpressions }}
        {{- $_ := set (index $matchExpressions 0) "values" (list $name) }}
        {{- toYaml $gpuAffinity | nindent 8 }}
      tolerations:
        {{- toYaml $.Values.global.gpuWorkload.tolerations | nindent 8 }}
      {{- else if $service.affinity }}
      affinity:
        {{- toYaml $service.affinity | nindent 8 }}
      {{- end }}
      {{- if and (not $service.useGPU) $service.tolerations }}
      tolerations:
        {{- toYaml $service.tolerations | nindent 8 }}
      {{- end }}
      containers:
      - name: {{ $name }}
        image: "{{ default $.Values.global.image.registry $service.image.registry }}/{{ $service.image.repository }}:{{ $service.image.tag }}"
        imagePullPolicy: {{ $.Values.global.image.pullPolicy }}
        {{- if $service.securityContext }}
        securityContext:
          {{- toYaml $service.securityContext | nindent 10 }}
        {{- end }}
        {{- if $service.livenessProbe }}
        livenessProbe:
          {{- toYaml $service.livenessProbe | nindent 10 }}
        {{- end }}
        {{- if $service.readinessProbe }}
        readinessProbe:
          {{- toYaml $service.readinessProbe | nindent 10 }}
        {{- end }}
        {{- if $service.command }}
        command:
        {{- toYaml $service.command | nindent 8 }}
        {{- end }}
        {{- if $service.args }}
        args:
        {{- toYaml $service.args | nindent 8 }}
        {{- end }}
        {{- if or $service.env $service.envFrom (and $service.useStandardEnv $.Values.common.standardEnv) }}
        {{- if $service.envFrom }}
        envFrom:
        {{- toYaml $service.envFrom | nindent 8 }}
        {{- end }}
        env:
        {{- if and $service.useStandardEnv $.Values.common.standardEnv }}
        {{- range $.Values.common.standardEnv }}
        - name: {{ .name }}
          value: {{ tpl .value $ }}
        {{- end }}
        {{- end }}
        {{- if $service.env }}
        {{- toYaml $service.env | nindent 8 }}
        {{- end }}
        {{- end }}
        {{- if $service.port }}
        ports:
        - containerPort: {{ $service.targetPort | default $service.port }}
        {{- end }}
        {{- if $service.useGPU }}
        resources:
          {{- toYaml $.Values.global.gpuWorkload.resources | nindent 10 }}
        {{- else if $service.resources }}
        resources:
          {{- toYaml $service.resources | nindent 10 }}
        {{- end }}
        {{- if $service.useGPU }}
        volumeMounts:
          {{- toYaml $.Values.global.gpuWorkload.volumeMounts | nindent 10 }}
      volumes:
        {{- toYaml $.Values.global.gpuWorkload.volumes | nindent 8 }}
        {{- end }}
{{- end }}
{{- end }}
{{- end -}}