{{- define "gpuorch.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{- define "gpuorch.labels" -}}
app.kubernetes.io/part-of: gpuorch
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "gpuorch.promAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "{{ . }}"
{{- end -}}
