### **Difference Between Elliptical and Symmetric Distributions**  

#### **1. Symmetric Distributions (General Case)**  
A distribution is **symmetric** if its probability density function (PDF) satisfies:  
$$
f(x - \mu) = f(\mu - x) \quad \text{for all } x.
$$  
- This means the distribution looks the same on both sides of its mean $\mu$.  
- Example: **Normal, Laplace, Logistic distributions** are symmetric.  

---

#### **2. Elliptical Distributions (More General Class)**  
An **elliptical distribution** is a **generalization of the multivariate normal distribution** where data is spread in an elliptical shape but may have heavier tails. It is defined by:  
$$
X \sim E(\mu, \Sigma, g)
$$  
where:
- $\mu$ is the mean vector.  
- $\Sigma$ is the covariance matrix (determining shape).  
- $g(r)$ is a generator function controlling tail behavior.  
- The density depends on the Mahalanobis distance:  
  $$
  f(x) = |\Sigma|^{-1/2} g\left( (x - \mu)^T \Sigma^{-1} (x - \mu) \right).
  $$  

- **Special Cases:**  
  - **Multivariate Normal**: $g(r) = e^{-r/2}$.  
  - **Multivariate t-distribution**: $g(r) = (1 + r/\nu)^{-(\nu + p)/2}$.  
  - **Cauchy, Logistic, and Kotz-type distributions** are also elliptical.  

---

### **Key Differences**  
| Feature | Symmetric Distributions | Elliptical Distributions |
|---------|--------------------------|----------------------------|
| **Definition** | A distribution that is mirror-image symmetric about its center. | A generalization of multivariate normality, defined by Mahalanobis distance. |
| **Multivariate Case?** | Not necessarily (e.g., a skewed bivariate distribution may still be symmetric along one axis). | Always multivariate, generalizing normality. |
| **Tails** | May be light-tailed (normal) or heavy-tailed (Laplace, Cauchy). | Can have **different tail behaviors** controlled by $g(r)$. |
| **Covariance Matrix** | Not always well-defined (e.g., Cauchy has infinite variance). | Always has a covariance matrix $\Sigma$ , but heavy tails affect its interpretation. |
| **Example** | Normal, Laplace, Uniform | Multivariate Normal, t-distribution, Cauchy |

---

### **Conclusion**  
All **elliptical distributions are symmetric**, but **not all symmetric distributions are elliptical**. Symmetric distributions can exist in **one dimension**, while elliptical distributions **only exist in multiple dimensions and generalize normality using Mahalanobis distance**.

