---
layout: default
title: matgl.apps.md
nav_exclude: true
---
# matgl.apps package

This package implements specific applications of matgl models. An example is their use for fitting interatomic
potentials parameterizing the potential energy surface (PES).


## matgl.apps.pes module

Implementation of Interatomic Potentials.


### _class_ matgl.apps.pes.Potential(model: nn.Module, data_mean: torch.tensor | None = None, data_std: torch.tensor | None = None, element_refs: np.ndarray | None = None, calc_forces: bool = True, calc_stresses: bool = True, calc_hessian: bool = False)
Bases: `Module`, [`IOMixIn`](matgl.utils.md#matgl.utils.io.IOMixIn)

A class representing an interatomic potential.

Initialize Potential from a model and elemental references.


* **Parameters**

    
    * **model** – Model for predicting energies.


    * **data_mean** – Mean of target.


    * **data_std** – Std dev of target.


    * **element_refs** – Element reference values for each element.


    * **calc_forces** – Enable force calculations.


    * **calc_stresses** – Enable stress calculations.


    * **calc_hessian** – Enable hessian calculations.



#### forward(g: dgl.DGLGraph, state_attr: torch.tensor | None = None, l_g: dgl.DGLGraph | None = None)

* **Parameters**

    
    * **g** – DGL graph


    * **state_attr** – State attrs


    * **l_g** – Line graph.



* **Returns**

    torch.tensor



* **Return type**

    energies, forces, stresses, hessian



#### training(_: boo_ )