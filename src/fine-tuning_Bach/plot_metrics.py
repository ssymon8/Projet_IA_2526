import json
import matplotlib.pyplot as plt
from pathlib import Path

# Remplace par ton hash
XP = "augmented-113e"
history_path = Path.home() / f"Projet_IA_2526/demucs/outputs/xps/{XP}/history.json"

history = json.load(open(history_path))

epochs          = list(range(1, len(history) + 1))
train_loss      = [h['train']['loss'] for h in history]
valid_loss      = [h['valid']['loss'] for h in history]
nsdr            = [h['valid']['nsdr'] for h in history]
nsdr_bassoon    = [h['valid']['nsdr_bassoon'] for h in history]
nsdr_clarinet   = [h['valid']['nsdr_clarinet'] for h in history]
nsdr_saxophone  = [h['valid']['nsdr_saxophone'] for h in history]
nsdr_violin     = [h['valid']['nsdr_violin'] for h in history]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

ax1.plot(epochs, train_loss, label='Train Loss', color='blue')
ax1.plot(epochs, valid_loss, label='Valid Loss', color='orange')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Evolution de la Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, nsdr,          label='Nsdr moyen',   color='black', linewidth=2)
ax2.plot(epochs, nsdr_bassoon,  label='Bassoon',      color='blue',   linestyle='--')
ax2.plot(epochs, nsdr_clarinet, label='Clarinet',     color='orange', linestyle='--')
ax2.plot(epochs, nsdr_saxophone,label='Saxophone',    color='green',  linestyle='--')
ax2.plot(epochs, nsdr_violin,   label='Violin',       color='red',    linestyle='--')
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Nsdr (dB)')
ax2.set_title('Evolution du Nsdr par instrument')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves1.png', dpi=150)
print("Graphique sauvegardé dans training_curves1.png")