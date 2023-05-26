# IHMPC-Lab: Laboratory for Infinite Horizon Model Predictive Controllers

---

## 1 - Introduction
---

The IHMPC–Lab is a Python package to implement state-of-the-art stabilizing DMC-type MPC strategies of Odloak’s family of controllers. This control family focuses on infinite horizon stabilizing MPC controllers with the feasibility of the optimization problem at any time step. The main features are: (i) the stabilizing properties are enforced by slacked terminal constraints, which tackles infeasible optimization problem conditions, (ii) offset free control laws, (iii) application of a canonical state-space model based on the analytical form of the step response of the system, (iv) capability of integration of the control law with RTO (Real-Time Optimization) targets, (v) capability of assessing economic targets directly in the control law.

The IHMPC-Lab supports the design of control laws for systems containing the following combinations of open-loop poles:
+ Stable [1,2,3]
+ Stable and integrating [4,5,6]
+ Stable and unstable [7,8,9]

Additionally, the control laws designed by the IHMPC-Lab can have the following features:

+ a fixed output set-point, $\mathbf{y}_{\text{sp},k}$, [1,4,5]
+ zone control approach:
    + including input targets, $\mathbf{u}_{\text{target},k}$, evaluated by an RTO layer [2,5,6]
    + including the gradient of an economic function [9]
    + including the approximated gradient of an economic function [3,8]

The following optimization problem presents the general formulation of these controllers:
```math
\min_{\boldsymbol{\eta}_{\text{k}}} V_k =  
\sum_{j=0}^{\infty} ||\mathbf{y}(k+j|k)-\mathbf{y}_{\text{sp},k}-\boldsymbol{\delta}_{y,k}-\Gamma_{\text{k}}(\boldsymbol{\delta}_{\text{in},k}, \boldsymbol{\delta}_{\text{un},k})||^2_{\mathbf{Q}_y} + \sum_{j=0}^{m-1}||\Delta \mathbf{u}(k+j|k)||^2_{\mathbf{R}}
+V_{\delta,k}+V_{eco,k},
 ```
subject to:

```math
\mathbf{g}_{tc}(\mathbf{x}^{\text{s}}, \mathbf{x}^{\text{in}},\mathbf{x}^{\text{un}},\boldsymbol{y}_{\text{sp},k},\boldsymbol{\delta}_{\text{y},k},\boldsymbol{\delta}_{\text{in},k},\boldsymbol{\delta}_{\text{un},k}) = \mathbf{0},
```
```math
\mathbf{g}_{eco}(\mathbf{u}_{\text{target},k},\boldsymbol{\delta}_{\text{u},k}) = \mathbf{0},
```
```math
\Delta \mathbf{u}_{\text{min}}\leq\Delta \mathbf{u}(k+j|k)\leq\Delta \mathbf{u}_{\text{max}}, j = 0,1,\ldots,m-1,
 ```
```math
\mathbf{u}_{\text{min}}\leq \mathbf{u}(k-1)+\sum_{j=0}^{m-1} \Delta \mathbf{u}(k+j|k)\leq \mathbf{u}_{\text{max}}.
```
```math
\mathbf{y}_{\text{min}}\leq \mathbf{y}_{\text{sp},k}\leq\mathbf{y}_{\text{max}}, 
```
 $`aaaaaaaaaaa`$
 $`aaaaaaaaaaa`$
 
(assuming zone control) where $`\mathbf{y}(k+j|k)`$ is the output prediction of the model evaluated at time step  $`k+j`$ with information available from step $`k`$, $`\Delta \mathbf{u}`$ is the control action in the incremental form, $`\mathbf{x}^{\text{p}}`$ (p=s,in,un) are the states obtained from the Odloaks's canonical state-space, $`\boldsymbol{\delta}_{p,k}`$ (p=s,in,un) are slack variables applied to soften the terminal equality contraint 
 $`\mathbf{g}_{tc}`$. $`\mathbf{g}_{eco}`$ is a constraint included when input targes are applied. The bounds on the decision variables are definied by  $`\Delta \mathbf{u}_{\text{min}}`$, $`\Delta \mathbf{u}_{\text{max}}`$, $`\mathbf{u}_{\text{min}}`$, $`\mathbf{u}_{\text{max}}`$, $`\mathbf{y}_{\text{min}}`$, $`\mathbf{y}_{\text{max}}`$. $`V_{\delta,k}`$ is a term to penalize the slack usage. $`V_{eco,k}`$ is a term related to the economic formulation. $`\mathbf{Q}_y`$, $`\mathbf{R}`$ are weighting matrices.

The decision variable vector, $`\boldsymbol{\eta}_{\text{k}}`$, contains the control actions and other decision variables of each supported formulation. The matrix $`\Gamma_{\text{k}}`$ is included to guarantee a finite cost function. 

## 2 - General Structure
---
The IHMPC-Lab is composed of three main components:

+ **Model** converts linear models (transfer functions and state-space) to one of Odloak's canonical state-space formulations used by IHMPCLab's controllers. It is used for open-loop simulations, and stability analysis, and is a requirement for the IHMPC component.

+ **IHMPC** designs the stabilizing DMC-type MPC controller of the Odloak's family and it is used to calculate control actions. It utilizes an internal **Kalman** filter object for state estimations. 

+ **Closedloop** is an automated interface between a linear plant model and the **IHMPC** objects, managing the control closed-loop, i.e, feeding the IHMPC with the current output ''measurement'' and implementing the control action. This object stores relevant simulation data, control performance indicators, and using the auxiliary **PlotManager** object, plots the simulation results.

## 3 - Demonstrative Examples
---
The IHMPCLab has a select set of examples, located in **Examples folder**, to demonstrate the module's capabilities through closed-loop simulations using both systems and controllers available in research papers. The file named "run" consists of a code to run each of these examples and save their results (see "results" section). The "run" file includes short introductory comments on the system and the controller utilized for each simulation.

It is possible to exclude examples from running by converting their lines in the "run" file into comments by adding "#" at the beginning of the line.

In order to find the simulation utilized by the paper associated with this CodeOcean capsule, please refer to the script found at "/code/Examples/Unstable Systems/Optimizing Targets/Provided Inputs/CSTR.py". This particular example is called at line 84 of the "run" file.

# References

---
1. D. Odloak, Extended robust model predictive control, AIChE Journal 50 (8) (2004) 1824–1836. URL https://doi.org/10.1002/aic.10175

2. L. A. Alvarez, D. Odloak, Robust integration of real time optimization with linear model predictive control, Computers & Chemical Engineering 34 (12) (2010) 1937–1944.  URL http://dx.doi.org/10.1016/j.compchemeng.2010.06.017.

3. L. A. Alvarez, D. Odloak, Reduction of the qp-mpc cascade structure to a single layer mpc, Journal of Process Control 24 (10) (2014) 1627–1638. URL http://dx.doi.org/10.1016/j.jprocont.2014.08.008


4. M. A. Rodrigues, D. Odloak, An infinite horizon model predictive control for stable and integrating processes, Computers and Chemical Engineering 27 (8-9) (2003) 1113–1128.  URL http://dx.doi.org/10.1016/S0098-1354(03)00040-1

5. Carrapiço, O. & Odloak, Darci. (2005). A stable model predictive control for integrating process. Computers & Chemical Engineering. 29. 1089-1099. URL https://doi.org/10.1016/j.compchemeng.2004.11.008

6. A. Alvarez, E. M. Francischinelli, B. F. Santoro, D. Odloak, Stable Model Predictive Control for Integrating Systems with Optimizing Targets, Industrial & Engineering Chemistry Research 48 (20) (2009) 9141–9150.  URL http://dx.doi.org/10.1021/ie900400j.

7. M. A. F. Martins, D. Odloak, A robustly stabilizing model predictive control strategy of stable and unstable processes, Automatica 67 (2016) 132–143. URL http://dx.doi.org/10.1016/j.automatica.2016.01.046

8. D. D. Santana, M. A. F. Martins, D. Odloak, One-layer gradient-based MPC + RTO strategy for unstable processes: a case study of a CSTR system, Brazilian Journal of Chemical Engineering 37 (1) (2020) 173–188.  URL http://dx.doi.org/10.1007/s43153-020-00018-w.

9. D. D. Santana, Economic and distributed model predictive control for non-stable processes : stability , feasibility , and integration, Ph.D. thesis, Universidade Federal da Bahia (2020).

</font>
