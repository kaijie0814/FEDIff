# FEDIff
**<u>F<u>inite <u>E<u>lement for Neutron <u>Diff<u>usion**

## Overview
**FEDiff** is a research-oriented neutron diffusion solver implementing the **continuous Galerkin finite element method (CG-FEM)** using the **deal.II** finite element library.

The code aims to provide a flexible and extensible platform for studying **multigroup neutron diffusion**, adaptive refinement strategies, and advanced mesh–energy coupling algorithms.  
This project is developed as part of an **ongoing Master’s thesis nuclear engineering**.


## Development Status
This repository contains **active research code** under continuous development.
- Some components are **experimental**
- Certain parts may be **incomplete or not yet optimized**
- Comments and documentation are **actively being expanded**
- Code structure is **evolving as algorithms mature**

Despite this, the solver is functional and can run representative multigroup diffusion problems such as C5G7 and UO2-MOX benchmark problems.

**Status:** active development

**Code quality:** evolving

**Purpose:** research prototype

## Feature
This solver aims to support several features or methods:
1. Adaptive mesh refinement
2. Energy depenndent mesh (under development)
