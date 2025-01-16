Github Markdown does not support LaTeX very well!!! For a better compiled Markdown, refer to the [Jupyter Notebook file](4D_Var.ipynb).

# 4D-Var

## Derivation of Strong-Constraint 4D-Var Algorithm for Lorenz-63 model


### Strong-Constraint 4D-Var Overview

The idea of strong-constraint 4D-Var is that we want to find an *actual solution* to the ODE system given above that "best fits" the data. This approach carries the inherent assumption that the model is perfect if we have the perfect initial condition, which is a decent assumption in our case since we are running a synthetic trial with perturbed observations from this model to begin with.

In other words, we look to find the appropriate initial condition $\mathbf{x}_0 = \mathbf{x}(t_0)$ that produces a solution $\mathbf{x}(t)$ that is "closest" to the data over the entire time interval (where we cannot neglect system dynamics on each of these intervals, due to nonlinearity).

Formally, we look to find an initial condition $\mathbf{x}_0$ that minimizes the cost functional

$`
\begin{equation*}
  J(\mathbf{x}_0) = \frac{1}{2} (\mathbf{x}_0 - \mathbf{x}_0^b)^T (P_0^b)^{-1}(\mathbf{x}_0 - \mathbf{x}_0^b) + \frac{1}{2} \sum_{k=0}^{K} (\mathscr{H}_k(\mathbf{x}_k) - \mathbf{y}_k)^T R_k^{-1}(\mathscr{H}_k(\mathbf{x}_k) - \mathbf{y}_k)
\end{equation*}
`$

subject to

$`
\begin{equation*}
  \mathbf{x}_{k+1} = \mathscr{M}_{k+1}(\mathbf{x}_k), \hspace{5pt} k = 0, 1, \ldots, K-1
\end{equation*}
`$

where $\mathscr{M}_k$ and $\mathscr{H}_k$ are the full, nonlinear model ODE and observation operators, $P_0^b$ is the (known) covariance of the background guess for the initial state, and $R_k$ is the (known) covariance of the observation operator.

To minimize this cost functional, we take the gradient of $J$ with respect to the initial condition, $\mathbf{x}_0$.

Note that, given a perturbation $\delta \mathbf{x}_0$ of the initial condition, the first variation is going to have the form
$` \delta J = \left(\nabla_{\mathbf{x}_0} J\right)^T\delta \mathbf{x}_0 `$

Note that we can linearize our model about this perturbation to get a Tangent Linear Model (TLM). This will simplify our analysis by ignoring higher-order behavior of the perturbations, which is often unneccessary to get good results.

The perturbation of the initial condition is propogated through our model using the tangent linear equation

$`\delta \mathbf{x}_{k+1} = M_{k+1} \delta \mathbf{x}_k`$

where $M_{k+1}$ is the Jacobian matrix of $`\mathscr{M}_{k+1}`$, with partial derivatives of $`\mathbf{x}_{k+1}`$ with respect to $`\mathbf{x}_k`$.

Taking the first variation of the cost functional gives

$`\delta J = (\mathbf{x} - \mathbf{x}_0^b)^T(P_0^B)^{-1}\delta \mathbf{x}_0 + \sum_{k=0}^K(\mathscr{H}_k(\mathbf{x}_k) - \mathbf{y}_k)^T R_k^{-1} H_k \delta x_k`$

where $H_k$ is the Jacobian of $\mathscr{H}_k$, the nonlinear observation model at timestep $k$.

Note that we want to explicitly determine the perturbation of the cost functional with respect to the perturbation of the *initial condition*, and that $\delta \mathbf{x}_k$ depends on the perturbation of the intial condition by the relationship

$`\delta \mathbf{x}_{k+1} = M_{k+1}M_{k}\ldots M_{1}M_{0} \delta \mathbf{x}_0`$

Also note that, if we do not have an observation at every timestep that we increment our state, then the linearizations used in the cost functional must be propogated through time appropriately (as in the equation above, multiplying the correct number of Jacobian matrices at each timestep that we have incremented our state vector and not recieved an observation).

Putting this all together, we are going to solve for the minimizer of this cost functional by introducing adjoint state vectors $\mathbf{p}_k$ at each observation timestep $k = 0, 1, \ldots, K$. Since, by our Tangent Linear Model equation, $`\delta \mathbf{x}_k = M_k \delta \mathbf{x}_{k-1}`$, we have that $`\mathbf{p}_k^T (\delta \mathbf{x}_k - M_k \delta \mathbf{x}_{k-1}) =0`$, so adding terms like this to the cost functional does not change its value.

Hence, we now look to minimize the augmented cost functional

$`\delta J = (\mathbf{x} - \mathbf{x}_0^b)^T(P_0^B)^{-1}\delta \mathbf{x}_0 + \sum_{k=0}^K(\mathscr{H}_k(\mathbf{x}_k) - \mathbf{y}_k)^T R_k^{-1} H_k \delta x_k - \sum_{k=0}^K \mathbf{p}_k^T (\delta \mathbf{x}_k - M_k \delta \mathbf{x}_{k-1})`$

Rearranging this equation, exploiting the symmetry of the covariance matrices, and regrouping in terms of each $\delta \mathbf{x}_k$, we have that

$`
\begin{split}
  \delta J = {}& \left[(P_0^B)^{-1}(\mathbf{x}_0 - \mathbf{x}_0^b) + H_0^T R_0^{-1}(\mathscr{H}_0(\mathbf{x}_0) - \mathbf{y}_0) + M_0^T \mathbf{p}_1 \right]\delta \mathbf{x}_0 \\
  &+ \left[{\Large\sum}_{k=1}^{K-1}H_k^T R_k^{-1}(\mathscr{H}_k(\mathbf{x}_k) - \mathbf{y}_k) - \mathbf{p}_k + M_k^T \mathbf{p}_{k+1}\right] \delta \mathbf{x}_k \\
  &+ \left[H_K^T R_k^{-1} (\mathscr{H}_K(\mathbf{x}_K) - \mathbf{y}_K) - \mathbf{p}_K\right]\delta \mathbf{x}_K
\end{split}
`$

which is valid for any choice of the adjoint states $\mathbf{p}_k$.

So we are going to choose these states such that the only term that survives is $\delta \mathbf{x}_0$.

This means simply that

$`
\begin{align*}
  \mathbf{p}_K &= H_K^T \mathbf{R}_k^{-1} (\mathscr{H}_K(\mathbf{x}_k) - \mathbf{y}_k) \\
  \mathbf{p}_k &= H_k^T R_k^{-1}(\mathscr{H}_k(\mathbf{x}_k) - \mathbf{y}_k) + M_k^T \mathbf{p}_{k+1}, \hspace{20pt}k = K-1, K-2, \ldots, 1 \\
  \mathbf{p}_0 &= (P_0^B)^{-1}(\mathbf{x}_0 - \mathbf{x}_0^b) + H_0^T R_0^{-1}(\mathscr{H}_0(\mathbf{x}_0) - \mathbf{y}_0) + M_0^T \mathbf{p}_1
\end{align*}
`$

where we have found then that

$`\delta J = \mathbf{p}_0^T \delta \mathbf{x}_0`$

as desired.

So $\mathbf{p}_0$ is the desired gradient that we will use in an iterative gradient descent algorithm.

Note that $\mathbf{p}_0$ depends on all of the other $\mathbf{p}_k$ in an iterative way backwards from $\mathbf{p}_K$.

We define the weighted innovation vector as
$$d_k = H_k^T R_k^{-1}(\mathscr{H}_k(\mathbf{x}_k) - \mathbf{y}_k) $$

Using this notation, we can rewrite this gradient in terms of a gradient on the initial state/background and initial observation ($\delta J^b$), and a gradient on the remaining observations ($\delta J^o$) by fixing

$`\delta J = \delta J^b + \delta J^o`$

where

$`\delta J^b = (P_0^B)^{-1}(\mathbf{x}_0 - \mathbf{x}_0^b)`$

and

$`\delta J^o = d_0 + M_0^T(d_1 + M_1^T(d_2 + M_2^T(\ldots M_{K-1}^T(d_K)))) `$

Hence, the algorithm for 4D-Var amounts to iterating through the following steps:

1. Given a guess for the initial condition, integrate the solution to the ODE forward in time using an ODE solver (like `solve_ivp` provided in scipy) over a given set of time gridpoints. This gives us our $\mathbf{x}_k$.

2. Determine the needed Jacobian matrices $M_k$ and $H_k$ at each observation timestep, as well as the (weighted, nonlinear) innovations terms $d_k$ that depend on each of these.
  
   (Note that if we only have observations at some of the timesteps that we have states at, we need to take matrix products over the time interval between observations to get the correct Jacobians used in our cost functional)

3. Construct the gradient $\delta J^o$ by working backward in time from the last observation timestep to the first observation timestep, multiplying the right sums by the right Jacobians (working inside-out of the equation for $\delta J^o$ given above)

4. Construct the full gradient $\delta J$ by adding $\delta J_o$ to $\delta J^b$.

5. Perform some sort of gradient descent algorithm to find the analyzed initial condition $\mathbf{x}_0^a$ that minimizes the cost functional at this iteration.

6. Repeat steps 1-5 with this new initial condition. Repeat until some sort of convergence is achieved.

<br>

#### Introduction to the Physical Model

We are trying to assimilate data into the Lorenz-63 Model, which is a toy model from meterorology. The model is a first-order ODE system of the form

We wish to assimilate the Lorenz-63 equations by standard 4D-Var, where the Lorenz-63 equations are given by

\begin{align*}
    \frac{dx}{dt} &= -\sigma(x-y) \\
    \frac{dy}{dt} &= \rho x - y - xz \\
    \frac{dz}{dt} &= xy - \beta z
\end{align*}

where $\sigma = 10 $, $\rho = 28$, and $\beta = 8/3$.

This system is known to be chaotic, which makes it a good candidate for testing how well our 4D-Var algorithm performs on a set of data.

#### Data Source for Experimentation

Since this is a toy model, it is difficult to assimilate real-life datasets with this model. So we will use what the book calls a "twin experiment" or "synthetic run" to evaluate the performance of our algorithm.

In other words, the following process will be used to generate the observation data:

1. Create a "true" solution by giving an ODE solver a "background" initial condition, and then integrate the solution over a time interval. This will be the reference for how well our algorithm behaves.

2. Create "observations" by taking the model solution at various gridpoints (not necessarily at every one), and perturbing it by some mean-zero gaussian value with a given covariance matrix. (This assumes that the observation operator is the identity operator)

This process will give us a set of "true" solutions at each time gridpoint, and then a set of "observations" at a selection of these gridpoints.

\\

#### Application to the Lorenz-63 system

We have that the Tangent Linear Model/Jacobian for the Lorenz-63 system for going from state timestep $k$ to state timestep $k+1$ (assuming that the length of time between these timesteps is the constant $\Delta t$) is given by

$$ M_{k:k+1} = \begin{pmatrix}
1 - \sigma \Delta t & \sigma \Delta t & 0 \\
(\rho - z_k) \Delta t & 1 - \Delta t & -x_k \Delta t \\
y_k \Delta t & x_k \Delta t & 1 - \beta \Delta t
\end{pmatrix} $$

Our data assimilation method only has observed data at every other timestep where we run our model. So we actually need to multiply two of these matrices together to get the $M_k$ values used in the cost functional.

Also, by the data generation algorithm given above, we have that the observation operator is given by
$$\mathscr{H}_k(x_k,y_k,z_k) = (x_k,y_k,z_k)$$
which is just the identity opertaor, and thus,
$$H_k = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}$$
is the desired Jacobian used in our algorithm.
