{{- define "gpuorch.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{- define "gpuorch.labels" -}}
app.kubernetes.io/part-of: gpuorch
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- /* Control components belong on the CPU control-plane node (PRD §3), not on GPU laptops. Soft preference so
       single-node / kind setups still schedule. */ -}}
{{- define "gpuorch.controlPlaneAffinity" -}}
tolerations: [{key: node-role.kubernetes.io/control-plane, operator: Exists, effect: NoSchedule}]
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference: {matchExpressions: [{key: node-role.kubernetes.io/control-plane, operator: Exists}]}
      - weight: 50
        preference: {matchExpressions: [{key: gpuorch.io/real-gpu, operator: NotIn, values: ["true"]}]}
{{- end -}}

{{- define "gpuorch.promAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "{{ . }}"
{{- end -}}
