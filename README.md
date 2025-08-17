# flowsheet-surrogate-online

This repository provides an implementation of **surrogate-based flowsheet model calibration with online learning**.  
The framework demonstrates how surrogate models (neural networks) can be incrementally updated during calibration to maintain accuracy under evolving operating conditions, while reducing the number of expensive direct flowsheet simulations.

## Objective
The main goal is to show how **online learning strategies** (Always-one, Always-both, Conditional, No-replay) improve calibration efficiency by:
- Reducing the reliance on computationally heavy direct simulations,
- Preserving surrogate global accuracy through replay buffers,
- Adapting surrogate models to new operating conditions in real time.

## Contents
- Example flowsheet model setup  
- Surrogate training and online update workflow  
- Calibration workflow driver script  

## Reference
If you use this code in your research, please cite the related article:  
*B. Palotai, G. Kis, T. Chován, Á. Bárkányi, "Online learning supported surrogate-based flowsheet model maintenance", Digital Chemical Engineering, 2025 (under review).*

---

📌 Developed as part of research on **surrogate-based flowsheet model maintenance** for Digital Twins.
