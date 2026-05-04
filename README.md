# DoseRAD2026

Starter code for the DoseRAD2026 challenge.

The workspace currently has:

- `../Dataset/training`: photon training cases.
- `../Dataset/proton/training`: proton training cases.
- `../Dataset/photon`: currently empty.

Each case is expected to contain `image/ct.mha`, `image/mr.mha`, a case JSON file, and many dose MHA files. Photon dose files follow `Dose_B*_CP*.mha`; proton dose files follow `Dose_B*_R*_L*.mha`.

Pour notre méthode 2.5D, ça veut dire:

photon: sample = un B/CP, repère aligné sur gantry_angle, avec aperture MLC en canal.
proton: sample = un B/R/L, repère aligné sur le ray, avec énergie/layer et profondeur radiologique en canaux.





Pour le photon :

Canaux anatomiques:
   3D CT slices beam-aligned (mettre en densité): CT[z-2], CT[z-1], CT[z], CT[z+1], CT[z+2] en utilisant hu_to_density

Canaux aperture/MLC:
    aperture_mask : carte binaire de l’ouverture MLC

sortie Dose_B{B}_CP{CP}.mha resamplée dans le même repère beam-aligned


Pour le proton:

Canaux anatomiques:
    - 3D CT slices beam-aligned (mettre en densité): CT[z-2], CT[z-1], CT[z], CT[z+1], CT[z+2]

Scalaires en cartes constantes:
    energy_norm : énergie du layer/beamlet

sortie Dose_B{B}_R{R}_L{L}.mha resamplée dans le repère ray-aligned

Canaux profondeur:
    WET : profondeur radiologique cumulée le long du rayon (optionnel)

Peut être ensuite le faire par FilM??? 

resampler dans un repère ray-aligned,
centrer le crop sur l’axe du ray,
garder toujours la même convention spatiale.

convention : 
    z = direction normalisée du faisceau
    x = cross(up, z) avec up = (0,0,1) sauf si presque parallèle, alors up = (1,0,0)
    y = cross(z, x)

z = 0 au point d’entrée du faisceau dans le body mask
centrer x=y=0 sur l’axe du ray/beam.

2 mm isotrope pour photon 
et 1x1x3 pour proton


Calcul de tailles 3D guide par les doses:

- profondeur `z`: traversee du `body_mask` le long de l'axe beam/ray
- section transverse `x/y`: soit bbox du support de dose seuillee, soit plus petit crop carre centre sur l'axe contenant une fraction cible de la dose integree
- marges explicites en mm
- recommandation finale aux percentiles `p95/p99/max`

Important:
- pour **calibrer les tailles**, le script ne resample pas le CT ou la dose dans un nouveau volume aligne
- il projette les centres de voxels dans le repere beam/ray-aligned, ce qui suffit pour mesurer les extents
- pour le **dataset d'entrainement**, en revanche, il faudra bien resampler CT et dose dans ce repere

Script:

```bash
/home/lyh/.venvs/doserad2026/bin/python scripts/recommend_crop_sizes.py \
  --dataset-root ../Dataset \
  --photon-spacing 2 2 2 \
  --proton-spacing 1 1 3 \
  --extent-mode energy \
  --energy-fraction 0.95 \
  --photon-cp-samples 5 \
  --proton-ray-samples 3 \
  --dose-rel-threshold 0.0001 \
  --progress-every 200 \
  --save-every 200 \
  --record-file recommend_crop_sizes_records.jsonl \
  --summary-file recommend_crop_sizes_summary.json
```

Le script prend:
- `z` depuis la traversee anatomique du beam/ray
- `x/y` depuis le volume de dose utile
- et propose des tailles voxellisees arrondies a un multiple de `16`

En mode `energy`:
- le crop est **centre sur l'axe** du beam/ray
- la section transverse est un **carre** `2r x 2r`
- le script cherche le plus petit volume de ce type contenant `95%` de la dose integree de la contribution

Robustesse du script:
- `--record-file` ecrit un enregistrement JSONL par contribution traitee
- `--summary-file` maintient un resume intermediaire des stats
- `--resume` reprend un calcul interrompu en sautant les contributions deja traitees
- `--progress-every N` affiche la progression et une ETA reguliere
- `--save-every N` force une sauvegarde partielle reguliere

Commande type pour reprendre un gros run:

```bash
/home/lyh/.venvs/doserad2026/bin/python scripts/recommend_crop_sizes.py \
  --dataset-root ../Dataset \
  --photon-spacing 2 2 2 \
  --proton-spacing 1 1 3 \
  --extent-mode energy \
  --energy-fraction 0.95 \
  --photon-cp-samples 5 \
  --proton-ray-samples 3 \
  --dose-rel-threshold 0.0001 \
  --progress-every 200 \
  --save-every 200 \
  --record-file recommend_crop_sizes_records.jsonl \
  --summary-file recommend_crop_sizes_summary.json \
  --resume
```

Remarque pratique:
- le mode `energy` est nettement plus couteux que `support`
- `--proton-ray-samples 20` revient en pratique a prendre tous les rays proton
- un seuil tres bas comme `1e-5` garde beaucoup plus de voxels et ralentit fortement le calcul


Photon
  samples: 1125
  body length mm    p95=370.0  p99=412.8  max=447.0
  dose depth mm     p95=382.0  p99=420.0  max=458.0
  half x mm         p95=100.0  p99=112.0  max=118.0
  half y mm         p95=100.0  p99=112.0  max=118.0
  recommended mm    x=244.0  y=244.0  z=440.0
  recommended vox   x=128  y=128  z=224
  square xy vox     128 x 128 x 224

Proton
  samples: 16172
  body length mm    p95=366.0  p99=407.0  max=456.0
  dose depth mm     p95=273.0  p99=321.0  max=393.0
  half x mm         p95=24.0  p99=27.0  max=33.0
  half y mm         p95=24.0  p99=27.0  max=33.0
  recommended mm    x=74.0  y=74.0  z=427.0
  recommended vox   x=80  y=80  z=144
  square xy vox     128 x 128 x 144