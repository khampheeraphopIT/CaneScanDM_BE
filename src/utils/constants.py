DISEASE_LABELS = ['Healthy', 'Mosaic', 'Notsugarcane', 'Redrot', 'Rust', 'Yellow']

label_map = {name: i for i, name in enumerate(DISEASE_LABELS)}
reverse_label_map = {i: name for i, name in enumerate(DISEASE_LABELS)}

