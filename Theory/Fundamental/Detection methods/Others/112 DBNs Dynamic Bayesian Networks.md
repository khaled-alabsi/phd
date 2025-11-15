### Dynamic Bayesian Networks (DBNs)
 Dynamic Bayesian Networks (DBNs) are an extension of Bayesian networks, designed to handle temporal processes by modeling the probabilistic dependencies over time. In the context of control charts for multiple variables, DBNs offer a way to monitor complex, interdependent process variables while considering their temporal evolution.

### Key Points in Using DBNs for Multivariate Control Charts:

1. **Temporal Dependencies**: DBNs model how a variable's state at one timestamp depends on its state at previous timestamps, capturing shifts, trends, or cyclical patterns over time.

2. **Handling Multiple Variables**: DBNs support multivariate analysis by accounting for dependencies between multiple features in your process data. For example, if feature    $X$    is related to feature    $Y$   , DBNs can model that relationship dynamically over time.

3. **Probabilistic Anomaly Detection**: By learning normal patterns in multivariate time series, DBNs can help identify deviations indicative of an out-of-control process. Anomalies are flagged when observed patterns deviate from the learned temporal dependencies.

4. **Adaptive Monitoring**: Unlike static models, DBNs adapt to changing conditions by updating conditional probabilities over time, which is helpful for detecting gradual drifts and shifts.

5. **Interpretability**: DBNs allow for visualizing relationships and dependencies between process variables, making it easier to pinpoint which variables influence anomalies when control limits are breached.

This makes DBNs a powerful tool in multivariate control chart applications, especially when temporal and inter-variable relationships are essential for detecting complex anomalies in manufacturing processes.

"Bayesian" generaly refers to a mathematical approach and philosophy rooted in Bayes' theorem, which calculates conditional probabilities. In the context of statistics and machine learning, it represents a method type, specifically Bayesian inference, that relies on updating beliefs (probabilities) based on evidence.

### Key Aspects of Bayesian Meaning and Methodology:

1. **Bayes’ Theorem**: The mathematical foundation is Bayes' theorem, which describes the probability of an event based on prior knowledge of related conditions. For events    $A$    and    $B$   , Bayes' theorem states:
   
   $$
   P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
   $$

2. **Bayesian Inference**: This method updates the probability estimate for a hypothesis as more evidence or data becomes available. In practice, it allows incorporating prior knowledge (a prior probability distribution) and adjusting it in light of new data to get a posterior probability.

3. **Applications Across Fields**: Bayesian methods are widely applied in fields like machine learning, econometrics, and control theory. They help create models that evolve as more data is gathered, making them highly adaptable and particularly useful for sequential data, like time series.

4. **Philosophical Approach**: Beyond just a method, Bayesian reasoning represents a philosophical approach to probability, treating probabilities as degrees of belief rather than fixed frequencies.

In summary, "Bayesian" signifies both a foundational mathematical principle and a type of inference methodology that’s key in probabilistic modeling and decision-making.

Let's set up a table to track the states of    $X$    (Temperature) and    $Y$    (Pressure) over two time steps, showing the values and probabilities used in each step.

### Values and Probabilities for Each State

| Time Step    $t$    |    $X$    (Temperature) State |    $Y$    (Pressure) State | Probability Calculation                                            | Result     |
|-----------------|--------------------------|------------------------|---------------------------------------------------------------------|------------|
|    $t = 1$          |    $X_1 = \text{high}$       |    $(Y_1 = \text{high})$     |    $(P(X_1 = \text{high}) \cdot P(Y_1 = \text{high})$    |    $(X_1 = \text{high})$    |   $0.6 \cdot 0.7 = 0.42$   |
|   $t = 1$   |    $X_1 = \text{high}$   |   $Y_1 = \text{low}$    |    $P(X_1 = \text{high}) \cdot P(Y_1 = \text{low})$    |    $(X_1 = \text{high})$    |   $0.6 \cdot 0.3 = 0.18$   |
|   $t = 1$   |   $X_1 = \text{low}$   |   $Y_1 = \text{high}$   |   $P(X_1 = \text{low}) \cdot P(Y_1 = \text{high})$    |    $(X_1 = \text{low})$   |   $0.4 \cdot 0.2 = 0.08$   |
|   $t = 1$   |   $X_1 = \text{low}$   |   $Y_1 = \text{low}$   |   $P(X_1 = \text{low}) \cdot P(Y_1 = \text{low})$    |    $(X_1 = \text{low})$   |   $0.4 \cdot$   0.8 = 0.32   $ |

At $t = 1$ , we calculate the joint probability of each state for $X_1$ and $Y_1$ as shown above.

---

### Transition Probabilities from $t = 1$ to $t = 2$ Now, let’s compute the probability of each state at $t = 2$ given the state at $t = 1$. We’ll look at each possible initial state from $t = 1$ , applying the transition probabilities for $X$ and $Y$ as defined.

---

#### Step-by-Step Probability Calculation for $P(X_2 = \text{high}, Y_2 = \text{high})$ 1. **Case 1: $X_1 = \text{high}, Y_1 = \text{high}$**

   - **Probability of initial state**: $P(X_1 = \text{high}) \cdot P(Y_1 = \text{high} | X_1 = \text{high}) = 0.6 \cdot 0.7 = 0.42$ - **Transition probabilities for $X_2$ and $Y_2$**:
     $$
     P(X_2 = \text{high} | X_1 = \text{high}) = 0.8
     $$
     $$
     P(Y_2 = \text{high} | Y_1 = \text{high}) = 0.9
     $$
   - **Combined probability for $t = 2$**:
     $$
     0.42 \cdot 0.8 \cdot 0.9 = 0.3024
     $$

2. **Case 2: $X_1 = \text{high}, Y_1 = \text{low}$**

   - **Probability of initial state**: $P(X_1 = \text{high}) \cdot P(Y_1 = \text{low} | X_1 = \text{high}) = 0.6 \cdot 0.3 = 0.18$
   - **Transition probabilities for $X_2$ and $Y_2$**:
     $$
     P(X_2 = \text{high} | X_1 = \text{high}) = 0.8
     $$
     $$
     P(Y_2 = \text{high} | Y_1 = \text{low}) = 0.4
     $$
   - **Combined probability for $t = 2$**:
     $$
     0.18 \cdot 0.8 \cdot 0.4 = 0.0576
     $$

3. **Case 3: $X_1 = \text{low}, Y_1 = \text{high}$**

   - **Probability of initial state**: $P(X_1 = \text{low}) \cdot P(Y_1 = \text{high} | X_1 = \text{low}) = 0.4 \cdot 0.2 = 0.08$
   - **Transition probabilities for $X_2$ and $Y_2$**:
     $$
     P(X_2 = \text{high} | X_1 = \text{low}) = 0.3
     $$
     $$
     P(Y_2 = \text{high} | Y_1 = \text{high}) = 0.9
     $$
   - **Combined probability for $t = 2$**:
     $$
     0.08 \cdot 0.3 \cdot 0.9 = 0.0216
     $$

4. **Case 4: $X_1 = \text{low}, Y_1 = \text{low}$**

   - **Probability of initial state**: $P(X_1 = \text{low}) \cdot P(Y_1 = \text{low} | X_1 = \text{low}) = 0.4 \cdot 0.8 = 0.32$
   - **Transition probabilities for $X_2$ and $Y_2$**:
     $$
     P(X_2 = \text{high} | X_1 = \text{low}) = 0.3
     $$
     $$
     P(Y_2 = \text{high} | Y_1 = \text{low}) = 0.4
     $$
   - **Combined probability for $t = 2$**:
     $$
     0.32 \cdot 0.3 \cdot 0.4 = 0.0384
     $$

---

### Final Probability Summation

To get the final probability    $P(X_2 = \text{high}, Y_2 = \text{high})$   , we add up the probabilities from each of these cases:

$$
P(X_2 = \text{high}, Y_2 = \text{high}) = 0.3024 + 0.0576 + 0.0216 + 0.0384 = 0.42
$$

This is how we calculate    $P(X_2 = \text{high}, Y_2 = \text{high})$    using the DBN model with specified initial and transition probabilities.

If the variables in a Dynamic Bayesian Network (DBN) are continuous (e.g., floats), the approach to probability calculation changes significantly. Instead of using discrete conditional probabilities, we typically employ **probability density functions** (PDFs) and **continuous distributions** to model relationships and transitions between states.

Here's how DBNs handle continuous variables:

### 1. **Continuous Distributions**
   - Each variable is represented by a probability distribution rather than discrete states. Common choices for continuous DBNs include the **Gaussian (Normal) distribution** and **Mixture of Gaussians**.
   - For instance, a variable    $X$    (e.g., temperature) at time    $t$    might follow a Gaussian distribution:
     $$
     X_t \sim \mathcal{N}(\mu, \sigma^2)
     $$
   - Parameters    $\mu$    (mean) and    $\sigma^2$    (variance) describe the distribution.

### 2. **Conditional Dependence via Linear Gaussian Models**
   - Continuous DBNs often assume **linear Gaussian relationships** between variables over time. If    $X_t$    depends on    $X_{t-1}$   , this relationship might be modeled as:
     $$
     X_t = a \cdot X_{t-1} + b + \epsilon
     $$
     where    $a$    and    $b$    are constants, and    $\epsilon \sim \mathcal{N}(0, \sigma^2)$    is Gaussian noise.

### 3. **Joint Probability Calculation with PDFs**
   - Rather than calculating exact probabilities, we work with joint PDFs for continuous variables.
   - For example, to calculate the probability of observing specific values of    $X_2$    and    $Y_2$   , you would integrate over the joint PDF of    $X_1$    and    $Y_1$   , weighted by their respective transition PDFs.

### Numerical Example: Two Continuous Variables Over Time

Let’s go through a simple example with two continuous variables,    $X$    (temperature) and    $Y$    (pressure), assuming Gaussian distributions.

1. **Define Initial Distributions** at    $t = 1$   :
   -    $X_1 \sim \mathcal{N}(10, 1)$    (mean = 10, variance = 1)
   -    $Y_1 \sim \mathcal{N}(5, 2)$   

2. **Define Transition Model** (e.g., linear Gaussian) between time steps:
   -    $X_2 | X_1 \sim \mathcal{N}(0.5 \cdot X_1 + 1, 1)$   
   -    $Y_2 | Y_1 \sim \mathcal{N}(0.7 \cdot Y_1 + 0.5, 1.5)$   

3. **Calculate Probabilities for Given Observations**:
   - To compute    $P(X_2 = 11, Y_2 = 6)$   , you would integrate over all possible values of    $X_1$    and    $Y_1$    using the joint PDF:
     $$
     P(X_2 = 11, Y_2 = 6) = \int_{x_1} \int_{y_1} P(X_2 = 11 | X_1 = x_1) P(Y_2 = 6 | Y_1 = y_1) P(X_1 = x_1) P(Y_1 = y_1) \, dx_1 \, dy_1
     $$

In practice, this integration is often performed using **sampling methods** like **particle filtering** or **Kalman filtering** (for linear Gaussian systems) to approximate the probabilities without solving the integrals analytically.

Let’s consider a real-world example of using a Dynamic Bayesian Network (DBN) to monitor a **chemical reactor** in a manufacturing plant. In this case, we want to continuously monitor three key process variables to detect anomalies and ensure that the process remains within control limits.

### Scenario: Chemical Reactor Process Monitoring

In this chemical reactor example, we monitor:
1. **Temperature** (   $T$   ) of the reactor
2. **Pressure** (   $P$   ) inside the reactor
3. **Concentration** (   $C$   ) of a reactant

Each of these variables is continuously measured and influences each other over time. For example:
- Temperature affects the pressure (higher temperatures typically increase pressure).
- Concentration is influenced by both temperature (affecting reaction speed) and pressure (affecting reactant solubility).

### DBN Setup with Continuous Variables

1. **Define Variables**:
   - **$T_t$**: Temperature at time $t$ (measured in degrees Celsius).
   - **$P_t$**: Pressure at time $t$ (measured in kPa).
   - **$C_t$**: Concentration of reactant at time $t$ (measured in mol/L).

2. **Define Initial Distributions** (assume these are known or estimated based on historical data):
   - $T_1 \sim \mathcal{N}(100, 5^2)$ (temperature starts at 100°C with a standard deviation of 5°C).
   - $P_1 \sim \mathcal{N}(200, 20^2)$ (pressure starts at 200 kPa with a standard deviation of 20 kPa).
   - $C_1 \sim \mathcal{N}(0.5, 0.05^2)$ (concentration starts at 0.5 mol/L with a standard deviation of 0.05 mol/L).

3. **Define Transition Models**:
   - **Temperature Transition**: Temperature at the next time step depends on the current temperature and pressure. For instance:
     $$
     T_{t+1} | T_t, P_t \sim \mathcal{N}(0.9 \cdot T_t + 0.05 \cdot P_t + 2, 3^2)
     $$
   - **Pressure Transition**: Pressure depends on both the temperature and concentration:
     $$
     P_{t+1} | P_t, T_t, C_t \sim \mathcal{N}(0.8 \cdot P_t + 0.1 \cdot T_t + 0.2 \cdot C_t + 5, 10^2)
     $$
   - **Concentration Transition**: Concentration depends on the current concentration and temperature:
     $$
     C_{t+1} | C_t, T_t \sim \mathcal{N}(0.95 \cdot C_t + 0.03 \cdot T_t, 0.02^2)
     $$

### Monitoring Process with the DBN

To monitor this process, we would use the DBN to predict the expected values of    $T$   ,    $P$   , and    $C$    at each time step and compare these with actual measurements to detect anomalies.

1. **Prediction Step**: Use the transition models to predict the distributions of    $T_{t+1}$   ,    $P_{t+1}$   , and    $C_{t+1}$    based on the current observed values of    $T_t$   ,    $P_t$   , and    $C_t$   .

2. **Update Step**: When new measurements come in, update the DBN based on the observed values to refine the predicted distributions for the next step.

3. **Anomaly Detection**: Compare observed values to the predicted ranges. If the observed value falls outside an acceptable range (e.g., beyond two standard deviations), flag it as a potential anomaly.

### Example Calculation

Suppose at time    $t = 1$   :
- Measured temperature    $T_1 = 98$   °C
- Measured pressure    $P_1 = 195$    kPa
- Measured concentration    $C_1 = 0.52$    mol/L

Using the transition models:
1. **Predict $T_2$**:
   $$
   T_2 \sim \mathcal{N}(0.9 \cdot 98 + 0.05 \cdot 195 + 2, 3^2) = \mathcal{N}(110.75, 9)
   $$
   Expected temperature at    $t = 2$    is 110.75°C with a variance of 9.

2. **Predict $P_2$**:
   $$
   P_2 \sim \mathcal{N}(0.8 \cdot 195 + 0.1 \cdot 98 + 0.2 \cdot 0.52 + 5, 10^2) = \mathcal{N}(171.904, 100)
   $$
   Expected pressure at    $t = 2$    is 171.904 kPa with a variance of 100.

3. **Predict $C_2$**:
   $$
   C_2 \sim \mathcal{N}(0.95 \cdot 0.52 + 0.03 \cdot 98, 0.02^2) = \mathcal{N}(3.019, 0.0004)
   $$
   Expected concentration at    $t = 2$    is 3.019 mol/L with a variance of 0.0004.

### Interpretation

At    $t = 2$   , we receive new measurements for temperature, pressure, and concentration. We compare these against the predicted means and variances:
- If temperature, pressure, or concentration deviates significantly (e.g., by more than two standard deviations from the mean), the DBN would flag it as an anomaly.
  
By modeling the dependencies and transitions among these three continuous variables, the DBN allows us to monitor the reactor process dynamically and detect potential out-of-control conditions before they escalate.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
time_steps = 50  # Number of time steps to simulate
t_mean, t_std = 100, 5  # Initial mean and std deviation for Temperature
p_mean, p_std = 200, 20  # Initial mean and std deviation for Pressure
c_mean, c_std = 0.5, 0.05  # Initial mean and std deviation for Concentration

# Transition model parameters
t_a, t_b, t_noise = 0.9, 0.05, 3  # Parameters for T transition
p_a, p_b, p_c, p_noise = 0.8, 0.1, 0.2, 10  # Parameters for P transition
c_a, c_b, c_noise = 0.95, 0.03, 0.02  # Parameters for C transition

# Initialize arrays to store simulated values
temperature = [np.random.normal(t_mean, t_std)]
pressure = [np.random.normal(p_mean, p_std)]
concentration = [np.random.normal(c_mean, c_std)]

# Simulate time series data
for t in range(1, time_steps):
    # Temperature at time t
    t_next = np.random.normal(t_a * temperature[-1] + t_b * pressure[-1] + 2, t_noise)
    temperature.append(t_next)
    
    # Pressure at time t
    p_next = np.random.normal(p_a * pressure[-1] + p_b * temperature[-1] + p_c * concentration[-1] + 5, p_noise)
    pressure.append(p_next)
    
    # Concentration at time t
    c_next = np.random.normal(c_a * concentration[-1] + c_b * temperature[-1], c_noise)
    concentration.append(c_next)

# Create a DataFrame for easier manipulation and plotting
data = pd.DataFrame({
    'Time': np.arange(time_steps),
    'Temperature': temperature,
    'Pressure': pressure,
    'Concentration': concentration
})

# Control limits (mean ± 2 * std for each variable)
t_control_limit = (np.mean(temperature), 2 * np.std(temperature))
p_control_limit = (np.mean(pressure), 2 * np.std(pressure))
c_control_limit = (np.mean(concentration), 2 * np.std(concentration))

# Identify out-of-control points
data['Temp_OutOfControl'] = (data['Temperature'] > t_control_limit[0] + t_control_limit[1]) | \
                            (data['Temperature'] < t_control_limit[0] - t_control_limit[1])
data['Press_OutOfControl'] = (data['Pressure'] > p_control_limit[0] + p_control_limit[1]) | \
                             (data['Pressure'] < p_control_limit[0] - p_control_limit[1])
data['Conc_OutOfControl'] = (data['Concentration'] > c_control_limit[0] + c_control_limit[1]) | \
                            (data['Concentration'] < c_control_limit[0] - c_control_limit[1])

# Plotting the control charts
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Temperature Control Chart
axs[0].plot(data['Time'], data['Temperature'], label='Temperature', color='blue')
axs[0].axhline(t_control_limit[0] + t_control_limit[1], color='red', linestyle='--', label='Control Limits')
axs[0].axhline(t_control_limit[0] - t_control_limit[1], color='red', linestyle='--')
axs[0].scatter(data['Time'][data['Temp_OutOfControl']], data['Temperature'][data['Temp_OutOfControl']], 
               color='orange', label='Out of Control')
axs[0].set_title('Temperature Control Chart')
axs[0].set_ylabel('Temperature (°C)')
axs[0].legend()

# Pressure Control Chart
axs[1].plot(data['Time'], data['Pressure'], label='Pressure', color='green')
axs[1].axhline(p_control_limit[0] + p_control_limit[1], color='red', linestyle='--', label='Control Limits')
axs[1].axhline(p_control_limit[0] - p_control_limit[1], color='red', linestyle='--')
axs[1].scatter(data['Time'][data['Press_OutOfControl']], data['Pressure'][data['Press_OutOfControl']], 
               color='orange', label='Out of Control')
axs[1].set_title('Pressure Control Chart')
axs[1].set_ylabel('Pressure (kPa)')
axs[1].legend()

# Concentration Control Chart
axs[2].plot(data['Time'], data['Concentration'], label='Concentration', color='purple')
axs[2].axhline(c_control_limit[0] + c_control_limit[1], color='red', linestyle='--', label='Control Limits')
axs[2].axhline(c_control_limit[0] - c_control_limit[1], color='red', linestyle='--')
axs[2].scatter(data['Time'][data['Conc_OutOfControl']], data['Concentration'][data['Conc_OutOfControl']], 
               color='orange', label='Out of Control')
axs[2].set_title('Concentration Control Chart')
axs[2].set_ylabel('Concentration (mol/L)')
axs[2].set_xlabel('Time')
axs[2].legend()

plt.tight_layout()
plt.show()

# Display DataFrame with control status for each variable
import ace_tools as tools; tools.display_dataframe_to_user(name="Simulated Reactor Process Data", dataframe=data)


```